"""证据筛选：把召回片段变成可直接支撑答案的材料。【原创增量 ② 的对照轴】

## 两种实现的差别不只是"贵不贵"

- :class:`LLMScreener` 对**每个**候选片段调用一次生成模型，让它同时产出一段
  面向问题的摘要与 1–10 的相关性评分。它贵，但摘要质量高——模型看过问题之后再
  决定"哪句话值得留下来"，这与"先按相关性排序、再统一摘要"是有本质差别的。
- :class:`CrossEncoderScreener` 先用交叉编码器（一个小模型、无需生成）把候选池
  从几十条收敛到十几条，只对这些片段调用生成模型写摘要。成本显著下降。

两者构成消融实验的核心对照：**LLM 重排相对交叉编码器的边际收益，究竟值多少 token**。
这个问题只有在本项目的架构下才问得出来——因为筛选被收敛成了单一协议，
两种做法可以互换而不触碰其他任何模块。

## 容错链为什么必须存在

筛选是**每个片段一次**的高频调用，因此是最容易遇到畸形输出的地方：
推理模型夹带思维链、模型把 JSON 包在代码围栏里、评分写成 ``"8/10"``、
甚至整段输出不是 JSON。这些都不是异常情况而是常态。一次解析失败只应导致
**该片段被丢弃**并计入 ``parse_failures``，绝不能中断整批——否则一个坏片段
就会让整次问答失败。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import anyio

from scitrace.config import ScreeningSettings
from scitrace.domain import (
    Citation,
    Evidence,
    Fragment,
    Source,
    SourceKey,
    render_inline_citation,
)
from scitrace.domain.evidence import MAX_RELEVANCE, MIN_RELEVANCE, make_evidence_key
from scitrace.domain.session import Usage
from scitrace.ports import LLMClient, LLMMessage
from scitrace.ports.reranker import Reranker
from scitrace.prompts import NOT_APPLICABLE, PromptSet
from scitrace.util.text import parse_json_object

logger = logging.getLogger(__name__)

__all__ = ["CrossEncoderScreener", "LLMScreener", "coerce_relevance"]

#: 评分字段的候选键名。模型的键名会漂移（``relevance`` / ``score`` / 带空格），
#: 与其在提示词里反复强调，不如在解析层认下来。
_SCORE_KEYS: tuple[str, ...] = ("relevance_score", "relevance", "score", "相关性评分", "评分")


def coerce_relevance(value: object) -> int:
    """把模型给出的评分归一化为 1–10 的整数。

    需要处理的实际形态：``8``、``"8"``、``8.0``、``"8/10"``、``"8 分"``、
    ``"评分：8"``。越界值钳制到边界而不是丢弃——一个 ``12`` 分说明模型认为它很相关，
    丢掉它反而是错的。

    Args:
        value: 模型给出的原始评分。

    Returns:
        归一化后的整数（1–10）；完全无法解析时返回 0，表示"这条不可用"。
    """
    if isinstance(value, bool):  # bool 是 int 的子类，必须先拦
        return 0
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "/" in text:
            numerator, _, denominator = text.partition("/")
            try:
                top, bottom = float(numerator.strip()), float(denominator.strip())
            except ValueError:
                return 0
            if bottom == 0:
                return 0
            number = top / bottom * MAX_RELEVANCE
        else:
            digits = "".join(
                char if (char.isdigit() or char == ".") else " " for char in text
            ).split()
            if not digits:
                return 0
            try:
                number = float(digits[0])
            except ValueError:
                return 0
    else:
        return 0

    rounded = round(number)
    return max(MIN_RELEVANCE, min(MAX_RELEVANCE, rounded))


def _extract_summary(payload: Mapping[str, object]) -> str:
    """从解析结果中取出摘要文本。"""
    for key in ("summary", "摘要", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # 解析成功但没有摘要字段：把整个对象里最长的字符串当作摘要，
    # 好过因为键名漂移就丢弃一条本来可用的证据。
    strings = [item for item in payload.values() if isinstance(item, str) and item.strip()]
    return max(strings, key=len).strip() if strings else ""


def _extract_score(payload: Mapping[str, object]) -> int:
    """从解析结果中取出评分。"""
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in _SCORE_KEYS:
        if key in lowered:
            return coerce_relevance(lowered[key])
    # 键名完全不匹配时，扫描所有值找一个像评分的
    for value in payload.values():
        score = coerce_relevance(value)
        if score > 0:
            return score
    return 0


class _ScreenerBase:
    """两种筛选实现共享的部分：引用渲染与证据构造。"""

    def __init__(
        self,
        *,
        prompts: PromptSet,
        settings: ScreeningSettings,
        max_evidence: int,
    ) -> None:
        self.prompts = prompts
        self.settings = settings
        self.max_evidence = max_evidence
        #: 最近一次 :meth:`screen` 的用量。调用方（会话构造）负责合并。
        self.last_usage = Usage()

    @property
    def name(self) -> str:
        """实现标识，用于实验记录与消融表。"""
        raise NotImplementedError

    def _citation_for(self, fragment: Fragment, sources: Mapping[SourceKey, Source]) -> Citation:
        """为片段构造引用信息。

        元数据缺失时用 ``unknown`` 而不是省略引用——**宁可显示"未知来源"，
        也不产生一个看起来像真的空引用**。后者会让读者以为已经核对过。
        """
        source = sources.get(fragment.source_key)
        stem = source.citation_stem if source is not None else "unknown"
        return Citation(
            evidence_key="",  # 由 :meth:`_build_evidence` 在构造 Evidence 时填入
            source_key=fragment.source_key,
            reference_key=stem,
            inline=render_inline_citation(stem, fragment.page_label),
            page_label=fragment.page_label,
        )

    def _build_evidence(
        self,
        fragment: Fragment,
        *,
        summary: str,
        relevance: int,
        sources: Mapping[SourceKey, Source],
    ) -> Evidence:
        citation = self._citation_for(fragment, sources)
        return Evidence(
            key=make_evidence_key(fragment.source_key, fragment.fragment_id),
            source_key=fragment.source_key,
            fragment_id=fragment.fragment_id,
            summary=summary,
            relevance=relevance,
            citation=citation.inline,
            page_label=fragment.page_label,
            section_path=list(fragment.section_path),
        )

    def _finalize(self, evidence: list[Evidence]) -> list[Evidence]:
        """过滤、排序、截断。

        过滤与排序放在基类，是为了让两种筛选实现产出的结果具有**同一种语义**：
        否则消融实验比较的就成了两套后处理，而不是两种筛选策略。
        """
        kept = [item for item in evidence if item.relevance >= self.settings.min_relevance]
        kept.sort(key=lambda item: (-item.relevance, item.key))
        return kept[: self.max_evidence]


class LLMScreener(_ScreenerBase):
    """对每个片段调用生成模型，产出摘要 + 相关性评分（SPEC §3.7 A）。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptSet,
        settings: ScreeningSettings,
        max_evidence: int,
    ) -> None:
        super().__init__(prompts=prompts, settings=settings, max_evidence=max_evidence)
        self.llm = llm

    @property
    def name(self) -> str:
        return "llm"

    async def screen(
        self,
        question: str,
        fragments: Sequence[Fragment],
        *,
        sources: Mapping[SourceKey, Source],
    ) -> list[Evidence]:
        """并发筛选全部候选片段。

        单个片段的任何失败（调用异常、输出无法解析）都只导致该片段被丢弃。
        """
        self.last_usage = Usage()
        if not fragments:
            return []

        limiter = anyio.Semaphore(self.settings.concurrency)
        lock = anyio.Lock()
        collected: list[Evidence] = []
        usage = Usage()

        async def work(fragment: Fragment) -> None:
            nonlocal usage
            async with limiter:
                citation = self._citation_for(fragment, sources)
                system, user = self.prompts.render_screening(
                    question=question,
                    citation=citation.inline,
                    text=fragment.text,
                )
                try:
                    response = await self.llm.complete(
                        [
                            LLMMessage(role="system", content=system),
                            LLMMessage(role="user", content=user),
                        ],
                        temperature=0.0,
                        json_object=True,
                    )
                except Exception as error:  # noqa: BLE001 - 单条失败不中断整批
                    logger.warning("筛选片段 %s 调用失败，已跳过：%s", fragment.fragment_id, error)
                    return

                payload = parse_json_object(response.content)
                async with lock:
                    usage = usage.merge(
                        Usage(
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                            estimated_cost=response.cost,
                            cost_currency=response.cost_currency,
                            cached_tokens=response.cached_tokens,
                            cost_known=response.cost_known,
                            llm_calls=1,
                            parse_failures=0 if payload is not None else 1,
                        )
                    )
                    if payload is None:
                        # 输出无法解析 → 丢弃该片段，但**不中断整批**。
                        # 这里刻意不构造一条 relevance=0 的占位证据：
                        # 它会被 _finalize 立刻过滤掉，白白制造一个
                        # summary 为空（进而触发校验失败）的中间对象。
                        logger.warning(
                            "筛选片段 %s 的输出无法解析为 JSON，已跳过：%r",
                            fragment.fragment_id,
                            response.content[:200],
                        )
                        return

                    summary = _extract_summary(payload)
                    relevance = _extract_score(payload)
                    if NOT_APPLICABLE in summary.upper():
                        relevance = 0
                    if not summary or NOT_APPLICABLE in summary.upper():
                        return
                    collected.append(
                        self._build_evidence(
                            fragment, summary=summary, relevance=relevance, sources=sources
                        )
                    )

        async with anyio.create_task_group() as task_group:
            for fragment in fragments:
                task_group.start_soon(work, fragment)

        self.last_usage = usage
        result = self._finalize(collected)
        logger.info(
            "筛选完成：%d 个候选 → %d 条证据（阈值 %d，解析失败 %d）",
            len(fragments),
            len(result),
            self.settings.min_relevance,
            usage.parse_failures,
        )
        return result

    async def aclose(self) -> None:
        """LLM 客户端由装配层管理，这里不关闭。"""
        return None


class CrossEncoderScreener(_ScreenerBase):
    """交叉编码器收敛候选后再调用模型写摘要（SPEC §3.7 B）。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        reranker: Reranker,
        prompts: PromptSet,
        settings: ScreeningSettings,
        max_evidence: int,
    ) -> None:
        super().__init__(prompts=prompts, settings=settings, max_evidence=max_evidence)
        self.llm = llm
        self.reranker = reranker

    @property
    def name(self) -> str:
        return "cross_encoder"

    async def screen(
        self,
        question: str,
        fragments: Sequence[Fragment],
        *,
        sources: Mapping[SourceKey, Source],
    ) -> list[Evidence]:
        """先重排取前 ``presummary_n``，再对这些片段写摘要（不要求模型打分）。

        评分直接取重排得分的分位映射——因为这一步的模型只被要求写摘要，
        让它再打分等于把"相关性判断"外包给一个没有全局视野的调用。
        """
        self.last_usage = Usage()
        if not fragments:
            return []

        ranked = await self.reranker.rerank(
            question, fragments, top_n=self.settings.presummary_n
        )
        if not ranked:
            return []

        limiter = anyio.Semaphore(self.settings.concurrency)
        lock = anyio.Lock()
        collected: list[Evidence] = []
        usage = Usage()
        total = len(ranked)

        async def work(position: int, fragment: Fragment) -> None:
            nonlocal usage
            async with limiter:
                citation = self._citation_for(fragment, sources)
                system, user = self.prompts.render_screening(
                    question=question, citation=citation.inline, text=fragment.text
                )
                try:
                    response = await self.llm.complete(
                        [
                            LLMMessage(role="system", content=system),
                            LLMMessage(role="user", content=user),
                        ],
                        temperature=0.0,
                        json_object=True,
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning("摘要片段 %s 失败，已跳过：%s", fragment.fragment_id, error)
                    return

                payload = parse_json_object(response.content)
                summary = _extract_summary(payload) if payload else ""
                parse_failed = payload is None
                if parse_failed:
                    # 交叉编码器路线下，摘要解析失败仍可用**重排名次**给分，
                    # 因为相关性判断来自重排器而非模型输出。
                    summary = response.content.strip()[:500] or "(输出无法解析)"
                if NOT_APPLICABLE in summary.upper():
                    return

                # 名次 → 评分的线性映射：第 1 名得满分，末位得阈值分。
                relevance = max(
                    self.settings.min_relevance,
                    round(MAX_RELEVANCE * (total - position + 1) / total),
                )
                async with lock:
                    usage = usage.merge(
                        Usage(
                            prompt_tokens=response.prompt_tokens,
                            completion_tokens=response.completion_tokens,
                            estimated_cost=response.cost,
                            cost_currency=response.cost_currency,
                            cached_tokens=response.cached_tokens,
                            cost_known=response.cost_known,
                            llm_calls=1,
                            parse_failures=1 if parse_failed else 0,
                        )
                    )
                    collected.append(
                        self._build_evidence(
                            fragment, summary=summary, relevance=relevance, sources=sources
                        )
                    )

        async with anyio.create_task_group() as task_group:
            for position, item in enumerate(ranked, start=1):
                task_group.start_soon(work, position, item.fragment)

        self.last_usage = usage
        return self._finalize(collected)

    async def aclose(self) -> None:
        """重排器由调用方管理，这里不关闭。"""
        return None

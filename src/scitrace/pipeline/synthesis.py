"""答案合成：把证据变成带可回溯引用的答案。

## 模块里唯一值得单独说明的设计

本模块把"调用模型"与"把模型输出变成可信答案"这两件事**分开**：

- :func:`bind_citations` 是**纯函数**，输入模型原文 + 证据表，输出规范化答案。
  它不碰网络、不碰模型，因此可以被穷举测试——而它恰恰是"可溯源"承诺真正落地的地方。
- :class:`AnswerSynthesizer` 只负责组装提示词、调用模型、把用量记下来。

这样拆分的理由很直接：如果把它们写在一起，"引用替换是否正确"就只能靠端到端测试
间接验证，而端到端测试跑一次要联网、要花钱、结果还不稳定（模型输出每次不同）。
实际情况是，引用后处理里有大量**只有构造畸形输入才测得到**的分支：
模型漏了圆括号、编造了不存在的键、把多个键用连字符连起来、
在正文里复述了材料中的 ``[ev-xxx]`` 方括号形式……

## 悬空引用为什么必须被剔除而不是保留

模型编造引用键是常见失败模式。若原样保留，读者会看到一个格式正确、
却指向不存在文献的引用——**这比没有引用更糟**，因为它看起来经过了核对。
因此 :func:`bind_citations` 会剥离所有无法解析到本次证据集的键并计数，
计数进入 ``Usage.dangling_citations``，理想值恒为 0。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from scitrace.config import AnswerSettings
from scitrace.domain import (
    Answer,
    Citation,
    Evidence,
    Source,
    SourceKey,
    render_bibtex,
)
from scitrace.domain.session import Usage
from scitrace.ports import LLMClient, LLMMessage
from scitrace.prompts import PromptSet, is_refusal

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerSynthesizer",
    "bind_citations",
    "count_dangling",
    "extract_evidence_keys",
]

#: 引用键。与 ``domain.evidence.EVIDENCE_KEY_PREFIX`` 保持一致，
#: 但这里用正则表达"8 位十六进制"的完整形态——只匹配前缀会把
#: ``ev-`` 后面跟着的任意文本（例如 ``ev-unknown``）也当成键。
_EVIDENCE_KEY_RE = re.compile(r"\bev-[0-9a-f]{8}\b")

#: 含引用键的圆括号组。用于把整组替换成渲染后的引用，
#: 而不是逐个键替换（后者会在 ``(ev-a, ev-b)`` 上产生嵌套括号）。
_CITATION_GROUP_RE = re.compile(r"\(([^()]*)\)")

#: 模型有时会照抄材料里的方括号形式 ``[ev-xxx]``。
_BRACKET_KEY_RE = re.compile(r"\[(ev-[0-9a-f]{8})\]")


def _strip_parens(text: str) -> str:
    """剥掉一层成对的圆括号（若存在）。"""
    stripped = text.strip()
    if stripped.startswith("(") and stripped.endswith(")") and stripped.count("(") == 1:
        return stripped[1:-1].strip()
    return stripped


def extract_evidence_keys(text: str) -> list[str]:
    """按出现顺序返回文本中的引用键（去重）。

    Args:
        text: 模型输出的答案正文。

    Returns:
        去重后的引用键列表，顺序为首次出现的顺序——
        参考文献列表要按引用先后排列，因此顺序必须保留。
    """
    seen: dict[str, None] = {}
    for match in _EVIDENCE_KEY_RE.finditer(text):
        seen.setdefault(match.group(0), None)
    return list(seen)


def bind_citations(
    raw_text: str,
    *,
    evidence: Sequence[Evidence],
    sources: Mapping[SourceKey, Source],
) -> Answer:
    """把模型输出中的引用键替换为可读文内引用，并生成参考文献表。

    处理四类实际出现过的输出形态：

    1. ``(ev-a1b2c3d4, ev-e5f6a7b8)`` —— 规范写法，整组替换；
    2. ``(ev-a1b2c3d4; ev-e5f6a7b8)`` / 连字符连接 —— 提示词禁止但模型仍会写，
       同样整组替换（**宽容接受**：格式不合规不是丢弃引用的理由）；
    3. ``[ev-a1b2c3d4]`` —— 模型照抄了材料里的方括号形式；
    4. 裸键 ``ev-a1b2c3d4`` —— 漏了括号。

    无法解析到本次证据集的键一律剥离并计入 ``dangling_citations``。

    Args:
        raw_text: 模型输出的答案正文。
        evidence: 本次进入上下文的证据（引用键的唯一合法集合）。
        sources: ``source_key -> Source``，用于渲染引用与 BibTeX。

    Returns:
        规范化后的 :class:`Answer`。若识别为拒答，则清空引用与参考文献——
        **拒答必须不带任何引用**，否则读者会以为它是基于证据得出的结论。
    """
    by_key = {item.key: item for item in evidence}

    if is_refusal(raw_text):
        return Answer(
            text=raw_text.strip(),
            raw_text=raw_text,
            refused=True,
            refusal_reason="模型判定证据不足以回答",
        )

    used_keys: list[str] = []
    dangling = 0
    for key in extract_evidence_keys(raw_text):
        if key in by_key:
            used_keys.append(key)
        else:
            dangling += 1
            logger.warning("答案中出现了无法解析的引用键 %s，已剥离", key)

    def inline_bare(key: str) -> str:
        """返回**不含**圆括号的文内引用。

        ``Evidence.citation`` 按 ``render_inline_citation`` 的约定是**含**括号的完整形式。
        把它直接塞进已有的括号组里会产生双括号（``((...))``），
        因此组内替换必须先剥掉外层括号，由组替换逻辑统一加一对。
        """
        item = by_key[key]
        if item.citation:
            return _strip_parens(item.citation)
        # 证据没带渲染好的引用（例如由外部构造）时现场渲染，避免输出空引用
        source = sources.get(item.source_key)
        stem = source.citation_stem if source is not None else "unknown"
        return f"{stem} {item.page_label}".strip()

    def inline_full(key: str) -> str:
        """返回**含**圆括号的完整引用，用于裸键与方括号形式的替换。"""
        return f"({inline_bare(key)})"

    # 1) 整组替换圆括号引用
    def replace_group(match: re.Match[str]) -> str:
        inner = match.group(1)
        keys = [key for key in extract_evidence_keys(inner) if key in by_key]
        if not keys:
            # 组内全是悬空键（或本来就没有键）：整组丢弃。
            # 保留会留下 "(ev-fake)" 这样格式正确却指向不存在文献的引用。
            return "" if _EVIDENCE_KEY_RE.search(inner) else match.group(0)
        return "(" + ", ".join(inline_bare(key) for key in keys) + ")"

    text = _CITATION_GROUP_RE.sub(replace_group, raw_text)

    # 2) 方括号形式（模型照抄了材料）
    text = _BRACKET_KEY_RE.sub(
        lambda match: inline_full(match.group(1)) if match.group(1) in by_key else "",
        text,
    )

    # 3) 剩余的裸键
    text = _EVIDENCE_KEY_RE.sub(
        lambda match: inline_full(match.group(0)) if match.group(0) in by_key else "",
        text,
    )

    text = _tidy(text)

    citations: list[Citation] = []
    references: dict[str, str] = {}
    for key in used_keys:
        item = by_key[key]
        source = sources.get(item.source_key)
        reference_key = source.citation_stem if source is not None else "unknown"
        citations.append(
            Citation(
                evidence_key=key,
                source_key=item.source_key,
                reference_key=reference_key,
                inline=inline_full(key),
                page_label=item.page_label,
            )
        )
        if reference_key not in references:
            if source is not None:
                references[reference_key] = render_bibtex(source)
            else:
                # 元数据缺失时仍生成一条最小条目：让读者明确知道"这里缺元数据"，
                # 而不是让参考文献表与正文引用数量对不上。
                references[reference_key] = _minimal_bibtex(reference_key)

    answer = Answer(
        text=text,
        raw_text=raw_text,
        citations=citations,
        references=references,
        refused=not citations,
        refusal_reason="" if citations else "答案未引用任何有效证据",
    )
    if dangling:
        logger.warning("本次答案剥离了 %d 处悬空引用", dangling)
    return answer


def _minimal_bibtex(reference_key: str) -> str:
    """为元数据缺失的来源生成最小 BibTeX 条目。"""
    title = "{Metadata unavailable}"
    note = "{scitrace could not resolve bibliographic metadata for this source}"
    return f"@misc{{{reference_key},\n  title = {title},\n  note = {note}\n}}"


def _tidy(text: str) -> str:
    """清理替换引用后留下的排版瑕疵。

    引用被剥离后常留下空括号、多余空格或句末的双重标点。不清掉的话，
    答案会呈现出"作者不修边幅"的观感，而这种观感会削弱读者对内容可信度的判断。
    """
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_dangling(raw_text: str, evidence: Sequence[Evidence]) -> int:
    """统计输出中无法解析到证据集的引用键数量。"""
    valid = {item.key for item in evidence}
    return sum(1 for key in extract_evidence_keys(raw_text) if key not in valid)


class AnswerSynthesizer:
    """组装材料、调用模型、绑定引用。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptSet,
        settings: AnswerSettings,
        max_evidence: int | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.settings = settings
        self.max_evidence = max_evidence if max_evidence is not None else settings.max_evidence
        #: 最近一次合成的用量。调用方（会话构造）负责合并。
        self.last_usage = Usage()

    def build_context(self, evidence: Sequence[Evidence]) -> str:
        """把证据渲染为送入模型的材料文本。

        条数上限在这里生效而不是在筛选层：筛选层的任务是"从候选中找出相关的"，
        而"上下文能装多少"是合成阶段的预算问题。混在一起会让
        ``screening.min_relevance`` 的语义变得含糊。
        """
        parts = [
            self.prompts.render_context_entry(
                evidence_key=item.key, citation=item.citation or "(unknown)", summary=item.summary
            )
            for item in evidence[: self.max_evidence]
        ]
        return "\n\n".join(parts)

    async def synthesize(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        sources: Mapping[SourceKey, Source],
        prior_answer: str | None = None,
    ) -> Answer:
        """生成答案。

        Args:
            question: 用户问题。
            evidence: 进入上下文的证据。
            sources: 文献元数据，用于渲染引用与参考文献。
            prior_answer: 上一轮答案（迭代场景）。给出时提示词会附加修订要求。

        Returns:
            规范化后的答案。**证据为空时不调用模型**，直接返回拒答——
            空上下文下让模型作答，等于请它凭记忆编造。
        """
        self.last_usage = Usage()
        if not evidence:
            logger.info("无证据可用，直接拒答（不调用模型）")
            return Answer(
                text="",
                raw_text="",
                refused=True,
                refusal_reason="没有检索到可用的证据",
            )

        system, user = self.prompts.render_synthesis(
            question=question,
            context=self.build_context(evidence),
            prior_answer=prior_answer,
        )
        response = await self.llm.complete(
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        self.last_usage = Usage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            estimated_cost_usd=response.cost_usd,
            llm_calls=1,
            dangling_citations=count_dangling(response.content, evidence),
        )

        answer = bind_citations(response.content, evidence=evidence, sources=sources)
        logger.info(
            "合成完成：引用 %d 处，参考文献 %d 条，拒答=%s",
            len(answer.citations),
            len(answer.references),
            answer.refused,
        )
        return answer

    async def aclose(self) -> None:
        """LLM 客户端由装配层管理，这里不关闭。"""
        return None

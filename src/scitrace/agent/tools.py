"""工具集：Agent 能做的事（SPEC §4.1）。

五个工具刻意**正交**：检索论文、收集证据、合成答案、重置范围、结束。
组合方式交给模型决定——这才是"Agentic"的意义所在。

一个刻意的合并：本项目**没有**把"关键词检索"与"语义检索"分成两个工具。
走哪几路召回是 `retrieval.strategy` 的配置问题，不是模型的决策问题：
让模型在两者之间选择，等于把一次可配置的工程决定变成了每次都要重新猜的赌博。
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from scitrace.agent.state import AgentState, Tool, ToolOutcome, status_line
from scitrace.pipeline.synthesis import SynthesisError
from scitrace.ports import ScoredFragment

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerQuestionTool",
    "FinishTool",
    "GatherEvidenceTool",
    "ResetScopeTool",
    "SearchLiteratureTool",
    "build_default_tools",
]


class _SearchArgs(BaseModel):
    query: str = Field(description="检索查询串。使用专业术语的原有形式，不要翻译缩写或方法名")
    year_from: int | None = Field(default=None, description="起始年份（含），不限定则省略")
    year_to: int | None = Field(default=None, description="结束年份（含），不限定则省略")


class SearchLiteratureTool(Tool):
    """在文献库中检索论文，确定后续取证的范围。"""

    name = "search_literature"
    description = (
        "在文献库中检索相关论文，确定候选论文集。返回候选论文清单与计数。"
        "后续的 gather_evidence 只会在这个范围内取证。"
        "如果结果不理想，可以换用更具体或更宽泛的查询再次调用。"
    )
    parameters_model = _SearchArgs

    def __init__(self, services) -> None:  # noqa: ANN001 - Services，避免循环导入
        self.services = services

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        assert isinstance(arguments, _SearchArgs)
        k = max(self.services.settings.retrieval.k, 10)
        results = await self.services.retriever.retrieve(arguments.query, k=k)
        results = self._filter_by_year(results, arguments.year_from, arguments.year_to)

        counts: dict[str, int] = {}
        for item in results:
            counts[item.fragment.source_key] = counts.get(item.fragment.source_key, 0) + 1

        state.scoped_source_keys = set(counts)
        state.total_paper_count = len(self.services.sources)
        state.relevant_paper_count = len(counts)
        if not counts:
            return ToolOutcome(
                observation=(
                    f"没有检索到与 {arguments.query!r} 相关的论文。"
                    "可以换用更宽泛的查询，或直接调用 finish 说明无法回答。"
                )
            )
        return ToolOutcome(observation=f"{self._render(results, counts)}\n{status_line(state)}")

    def _filter_by_year(
        self, results: list[ScoredFragment], year_from: int | None, year_to: int | None
    ) -> list[ScoredFragment]:
        if year_from is None and year_to is None:
            return results
        kept: list[ScoredFragment] = []
        for item in results:
            source = self.services.sources.get(item.fragment.source_key)
            year = source.year if source is not None else None
            if year is None:
                # 年份未知时**保留**而不是丢弃：丢弃会让"补全元数据失败"
                # 变成"这篇论文不存在"，而前者是可接受的降级。
                kept.append(item)
            elif (year_from is None or year >= year_from) and (year_to is None or year <= year_to):
                kept.append(item)
        return kept

    def _render(self, results: list[ScoredFragment], counts: dict[str, int]) -> str:
        lines = [f"检索到 {len(counts)} 篇候选论文："]
        for key, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]:
            source = self.services.sources.get(key)
            title = source.title if source is not None else "(元数据缺失)"
            year = source.year if source is not None and source.year else "年份未知"
            lines.append(f"- [{key}] {title}（{year}，命中 {count} 段）")
        return "\n".join(lines)


class _GatherArgs(BaseModel):
    question: str = Field(description="要为之收集证据的具体问题，可以与原问题不同")


class GatherEvidenceTool(Tool):
    """在候选论文范围内检索片段并筛选为证据。"""

    name = "gather_evidence"
    description = (
        "在候选论文范围内检索相关片段，并用模型筛选出真正有用的证据。"
        "返回新增证据条数与状态。若连续多次没有新增证据，说明该范围已经挖尽。"
    )
    parameters_model = _GatherArgs

    def __init__(self, services) -> None:  # noqa: ANN001
        self.services = services

    #: 零证据时的回退策略。
    #:
    #: 选 `hybrid_rrf` 而非 `hybrid_rrf_rerank`：后者要加载重排模型，
    #: 而回退只在少数失败查询上触发，为它常驻一个 Cross-Encoder 不划算。
    FALLBACK_STRATEGY = "hybrid_rrf"

    async def _fallback_retrieve(
        self, question: str, state: AgentState
    ) -> tuple[list, bool]:
        """主策略零证据时，换一条检索通路再试一次。

        **只筛真正新的片段**：主策略已经筛过（并拒绝）的片段在这里被排除，
        否则回退会把同一批片段用第二次调用重筛一遍——同一片段对同一问题的
        评分不会改变，那是纯浪费。这条约束让回退的额外开销只落在
        "换了通路才召回到的新片段"上。

        Args:
            question: 本轮取证用的子问题。
            state: 当前会话状态（读取已筛集合，写入新筛集合）。

        Returns:
            ``(新增证据, 是否真的执行了回退)``。未执行回退时第二项为 False。
        """
        try:
            results = await self.services.retriever.retrieve(
                question,
                allowed_keys=state.scoped_source_keys,
                strategy=self.FALLBACK_STRATEGY,
            )
        except Exception as error:  # noqa: BLE001 - 回退失败不该中断会话
            logger.warning("零证据回退检索失败，保持原结果：%s", error)
            return [], False

        fresh = [
            item.fragment
            for item in results
            if item.fragment.fragment_id not in state.screened_fragment_ids
        ]
        if not fresh:
            return [], True
        evidence = await self.services.screener.screen(
            question, fresh, sources=self.services.sources
        )
        state.screened_fragment_ids.update(item.fragment_id for item in fresh)
        return list(evidence), True

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        assert isinstance(arguments, _GatherArgs)
        results = await self.services.retriever.retrieve(
            arguments.question, allowed_keys=state.scoped_source_keys
        )
        # **在筛选之前**剔除已经入证的片段。
        #
        # 这不是微优化，是实测发现的主要成本来源：Agentic 模式一次真实运行的
        # 108 次 LLM 调用里，约 100 次是 gather_evidence 内部的筛选调用，
        # 而 agent 连调了 10 次 gather_evidence、每次都把同一批片段重新筛一遍。
        # 早先的去重发生在 `state.add_evidence`（筛选**之后**）——
        # 也就是说"这个片段我已经筛过了"这件事，系统要花完全额成本之后才知道。
        #
        # 实测该次运行的全部观测合计仅 2,603 token，而总消耗 182,866 ——
        # 成本几乎全在筛选调用上，不在历史重发上。所以省这里的收益最直接。
        # 排除两类片段：**已入证**的，以及**已筛过但被拒**的。
        # 后者同样重要——同一片段对同一问题的评分不会改变，重筛一次就是白花一次调用。
        known_fragments = {item.fragment_id for item in state.evidence}
        fragments = [
            item.fragment
            for item in results
            if item.fragment.fragment_id not in known_fragments
            and item.fragment.fragment_id not in state.screened_fragment_ids
        ]
        if not results:
            return ToolOutcome(
                observation=f"该范围内没有检索到片段。\n{status_line(state)}"
            )
        if not fragments:
            # 明确告诉 agent"这里已经挖尽了"，而不是让它再换个说法重试一次。
            # 早先它收到的是"新增 0 条证据"，这个信号太弱——实测它连续换了 5 种
            # 措辞重复调用同一个工具。
            return ToolOutcome(
                observation=(
                    "本次检索到的片段**全部已在证据集中**，没有新内容可筛。"
                    "继续用相同或相近的查询不会有新收获：请换一个明显不同的角度，"
                    "或直接用已有证据作答（answer_question）后结束（finish）。"
                    f"\n{status_line(state)}"
                )
            )
        evidence = await self.services.screener.screen(
            arguments.question, fragments, sources=self.services.sources
        )
        # 记账：这一批都已经筛过了，无论采纳与否
        state.screened_fragment_ids.update(item.fragment_id for item in fragments)
        added = state.add_evidence(evidence)
        self.services.merge_usage(self.services.screener.last_usage)

        # ---- 零证据回退（改进 C'）----
        #
        # 实测依据：某类问题（"哪家机构提出了 X"这种实体归属型）在 dense_mmr 下
        # 会把**正确文献排到第 3 名之后、片段相似度≈0**，筛选于是全部拒绝，
        # 最终零证据 → 拒答。这类拒答在一道实体归属题上占 14%（5/35 次）。
        # 而换成 hybrid_rrf 后，同一查询能把正确文献排到第 1 名（实验七 7.7）。
        #
        # 但**不能全局换策略**：实验二十一实测全局 hybrid_rrf 会把跨论文综合
        # 打出 −30pp（RRF 无 MMR 多样性，单词命中散落到多篇论文上）。
        # 所以只在"已经失败"时才切换——默认路径完全不变。
        if added == 0 and self.services.settings.agent.zero_evidence_fallback:
            fallback_evidence, fallback_used = await self._fallback_retrieve(
                arguments.question, state
            )
            if fallback_used:
                evidence = [*evidence, *fallback_evidence]
                added = state.add_evidence(fallback_evidence)
                self.services.merge_usage(self.services.screener.last_usage)

        summary = (
            f"新增 {added} 条证据（本次候选 {len(fragments)} 段，"
            f"筛选后 {len(evidence)} 条，累计 {len(state.evidence)} 条）。"
        )
        if added == 0:
            summary += "本轮没有新增证据，考虑换一个角度提问或直接作答。"
        else:
            summary += "\n" + "\n".join(
                f"- [{item.key}] {item.summary[:200]}" for item in evidence[:5]
            )
        return ToolOutcome(observation=f"{summary}\n{status_line(state)}")


class AnswerQuestionTool(Tool):
    """基于当前证据集合成答案。不结束会话。"""

    name = "answer_question"
    description = (
        "基于已收集的证据生成带引用的答案。这不会结束会话——"
        "你可以先用它看看当前证据能支撑出什么，再决定是否继续收集。"
    )

    def __init__(self, services) -> None:  # noqa: ANN001
        self.services = services

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        if not state.evidence:
            return ToolOutcome(
                observation="当前没有任何证据，无法作答。请先调用 gather_evidence。"
            )
        try:
            answer = await self.services.synthesizer.synthesize(
                state.question, state.evidence, sources=self.services.sources
            )
        except SynthesisError as error:
            # 合成**故障**与"模型说了证据不足"必须分开：前者是 FAIL，
            # 后者是正常的 REFUSED/UNSURE。混为一谈会把一个需要修的问题
            # 伪装成一个正确的行为。
            state.notes.append("synthesis_failed")
            logger.error("合成失败：%s", error)
            self.services.merge_usage(self.services.synthesizer.last_usage)
            raise
        self.services.merge_usage(self.services.synthesizer.last_usage)
        state.answer = answer
        if answer.refused:
            return ToolOutcome(
                observation=f"合成结果为拒答（{answer.refusal_reason}）。"
                "如需给出答案，请补充更多证据。"
            )
        return ToolOutcome(
            observation=f"已生成答案（{len(answer.citations)} 处引用）：\n{answer.text}"
        )


class ResetScopeTool(Tool):
    """清空候选论文集，保留已收集的证据。"""

    name = "reset_scope"
    description = "清空候选论文范围，下次 gather_evidence 会重新在全库检索。已收集的证据会保留。"

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        state.scoped_source_keys = None
        return ToolOutcome(
            observation=f"已重置检索范围（保留 {len(state.evidence)} 条已有证据）。"
        )


class _FinishArgs(BaseModel):
    has_answer: bool = Field(
        description="是否已经得出可回答问题的结论。证据不足时应为 false，而不是给出猜测"
    )
    reason: str = Field(default="", description="简短说明结束原因。不要在这里写答案正文")


class FinishTool(Tool):
    """结束会话。"""

    name = "finish"
    description = (
        "结束本次会话。若已得出的结论足以回答问题，has_answer 设为 true；"
        "若证据不足，设为 false 并说明原因——**明确说不知道比给出猜测更有价值**。"
    )
    parameters_model = _FinishArgs

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        assert isinstance(arguments, _FinishArgs)
        if arguments.reason:
            state.notes.append(f"finish: {arguments.reason}")
        return ToolOutcome(
            observation=f"会话结束（has_answer={arguments.has_answer}）。",
            stop=True,
            has_answer=arguments.has_answer,
        )


def build_default_tools(services) -> list[Tool]:  # noqa: ANN001 - Services
    """构造默认工具集。顺序即呈现给模型的顺序。"""
    return [
        SearchLiteratureTool(services),
        GatherEvidenceTool(services),
        AnswerQuestionTool(services),
        ResetScopeTool(),
        FinishTool(),
    ]

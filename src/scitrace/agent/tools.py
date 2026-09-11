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

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        assert isinstance(arguments, _GatherArgs)
        results = await self.services.retriever.retrieve(
            arguments.question, allowed_keys=state.scoped_source_keys
        )
        fragments = [item.fragment for item in results]
        if not fragments:
            return ToolOutcome(
                observation=f"该范围内没有检索到片段。\n{status_line(state)}"
            )
        evidence = await self.services.screener.screen(
            arguments.question, fragments, sources=self.services.sources
        )
        added = state.add_evidence(evidence)
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

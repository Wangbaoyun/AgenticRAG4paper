"""Agent 状态与工具集。

工具集刻意保持**小而正交**（SPEC §4.1）：五个工具各管一件事，
组合方式交给模型决定。工具过多会让模型在无关选项上浪费决策，
而工具语义重叠（例如同时有"按关键词搜"和"按语义搜"）会让它难以理解何时用哪个——
后者在本项目里被刻意合并成 `search_literature` 一个工具，
由配置的检索策略在内部决定走哪几路。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from scitrace.domain import Answer, Evidence, SourceKey
from scitrace.domain.session import ActionRecord
from scitrace.ports import ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["AgentState", "Tool", "ToolOutcome", "status_line"]


class AgentState(BaseModel):
    """一次 Agent 会话的可变状态。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    #: 候选论文集。``None`` 表示"尚未限定范围"（全库检索），
    #: 空集合表示"限定了范围但一篇都没找到"——这两者语义不同，不能合并。
    scoped_source_keys: set[SourceKey] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    answer: Answer | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    total_paper_count: int = 0
    relevant_paper_count: int = 0

    def add_evidence(self, items: Sequence[Evidence]) -> int:
        """按引用键去重后追加证据，返回**新增**条数。

        去重是必需的：多轮 `gather_evidence` 会反复召回同一片段，
        不去重会让上下文里塞满重复内容，浪费预算且稀释信号。
        """
        known = {item.key for item in self.evidence}
        fresh = [item for item in items if item.key not in known]
        self.evidence.extend(fresh)
        return len(fresh)


def status_line(state: AgentState, *, cost_usd: float = 0.0) -> str:
    """生成注入下一轮提示的状态串（SPEC §4.2）。

    状态串的作用常被低估：它让模型在不消耗额外调用的情况下知道
    "我已经找了多少篇、拿到了多少证据、花了多少钱"，
    从而自己决定是继续深挖还是收手。没有它，模型只能靠对话历史猜测进度。
    """
    papers = (
        "全部"
        if state.scoped_source_keys is None
        else f"{state.relevant_paper_count}/{state.total_paper_count}"
    )
    return f"papers={papers} evidence={len(state.evidence)} cost=${cost_usd:.4f}"


class ToolOutcome(BaseModel):
    """一次工具执行的结果。"""

    model_config = ConfigDict(extra="forbid")

    observation: str
    #: 是否应当结束会话。
    stop: bool = False
    #: ``finish`` 工具用来表达"有没有拿到答案"。
    has_answer: bool | None = None


class Tool:
    """工具基类。

    子类实现 :meth:`_run`；基类负责统一处理**参数非法**的情况——
    收到缺字段或类型错误的参数时返回一条说明错误的观测让模型自我纠正，
    而**不抛异常**。这很重要：模型偶尔给出非法参数是常态，
    为此中断整个会话等于把一次可恢复的小失误放大成一次失败的问答。
    """

    #: 工具名。会出现在提示词与日志里，改动等同于接口变更。
    name: str = ""
    #: 给模型看的功能说明。
    description: str = ""
    #: 参数模型（pydantic）；``None`` 表示无参数。
    parameters_model: type[BaseModel] | None = None

    def spec(self) -> ToolSpec:
        """导出为给模型的工具声明（JSON Schema 由 pydantic 生成）。"""
        schema: dict[str, object] = {"type": "object", "properties": {}}
        if self.parameters_model is not None:
            schema = self.parameters_model.model_json_schema()
        return ToolSpec(name=self.name, description=self.description, parameters=schema)

    async def run(self, arguments: dict[str, object], state: AgentState) -> ToolOutcome:
        """校验参数并执行。"""
        if self.parameters_model is None:
            validated = None
        else:
            try:
                validated = self.parameters_model.model_validate(arguments)
            except Exception as error:  # noqa: BLE001 - 参数错误应可恢复
                logger.info("工具 %s 收到非法参数：%s", self.name, error)
                return ToolOutcome(
                    observation=(
                        f"工具 {self.name} 的参数不合法：{error}。"
                        "请检查参数名与类型后重试。"
                    )
                )
        try:
            return await self._run(validated, state)
        except Exception as error:  # noqa: BLE001 - 工具失败不应中断整个会话
            logger.warning("工具 %s 执行失败：%s", self.name, error, exc_info=True)
            return ToolOutcome(observation=f"工具 {self.name} 执行失败：{error}")

    async def _run(self, arguments: BaseModel | None, state: AgentState) -> ToolOutcome:
        """子类实现的实际逻辑。"""
        raise NotImplementedError

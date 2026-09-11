"""跨端口共享的小型契约对象。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scitrace.domain import Fragment

__all__ = ["ScoredFragment", "ToolSpec"]


class ScoredFragment(BaseModel):
    """检索结果：片段 + 得分 + 来源路标识。

    三种检索后端（向量、全文、重排）返回同一种结构，使上游的融合、筛选与排序逻辑
    不必分支——这是"可插拔检索内核"（SPEC §9 增量 ②）能成立的前提。

    ``score`` 的**量纲随后端而异**（余弦相似度 ∈ [-1,1]、BM25 无上界、
    RRF 约 ∈ (0, 0.05]）。因此它只用于**同一路内的排序**；
    跨路融合必须走 RRF 这类基于**名次**的方法，不能直接相加或比较得分。
    """

    model_config = ConfigDict(extra="forbid")

    fragment: Fragment
    score: float
    #: 产生本结果的后端名（``"dense"`` / ``"bm25"`` / ``"rerank"``），用于排障与消融统计。
    origin: str = ""
    #: 名次（1-based），供 RRF 使用。检索后端填；未填时为 0。
    rank: int = Field(default=0, ge=0)


class ToolSpec(BaseModel):
    """提供给 LLM 的工具声明。

    ``parameters`` 是标准 JSON Schema，由 pydantic 模型导出——
    这样工具参数的类型定义只有一个来源（pydantic 模型），
    不会出现"文档里写的参数"与"代码里解析的参数"不一致的经典问题。
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, object] = Field(default_factory=dict)

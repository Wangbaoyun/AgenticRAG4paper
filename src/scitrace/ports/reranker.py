"""重排器契约 —— 可插拔检索内核的第三个可换部件。

与 :class:`~scitrace.ports.screener.EvidenceScreener` 的分工需要说清楚，
否则两者看起来在做同一件事：

- **Reranker**：对 ``(查询, 片段)`` 打一个相关性分，**只看相关性**，
  不做摘要、不调用生成模型。它便宜、快、可批量，用于把候选池从几十条收敛到几条。
- **Screener**：在重排之后的候选上调用生成模型，产出**可供合成使用的摘要**
  与 0–10 的评分。它贵、慢，但产出的是"能直接拼进上下文的材料"。

因此 ``hybrid_rrf_rerank`` 策略是"先重排收敛候选，再筛选生成摘要"，
两者串联而非二选一（SPEC §3.6 与 §3.7）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from scitrace.domain import Fragment
from scitrace.ports.common import ScoredFragment

__all__ = ["Reranker"]


@runtime_checkable
class Reranker(Protocol):
    """交叉编码器式的相关性重排。"""

    @property
    def model_name(self) -> str:
        """模型标识，用于实验记录与消融表。"""
        ...

    async def rerank(
        self, query: str, fragments: Sequence[Fragment], *, top_n: int
    ) -> list[ScoredFragment]:
        """按相关性对片段重排，返回前 ``top_n`` 条。

        Args:
            query: 原始查询串。
            fragments: 候选片段。
            top_n: 返回条数上限。

        Returns:
            按得分降序排列，``origin="rerank"``，``rank`` 从 1 开始。

        Note:
            ``score`` 的量纲由模型决定（交叉编码器通常输出 logits 或概率），
            因此**只在本次调用内部可比**，不可与向量相似度或 BM25 得分混用。
            这一点正是 SPEC §3.6 规定"跨路融合必须走基于名次的 RRF"的原因。
        """
        ...

    async def aclose(self) -> None:
        """释放模型句柄等资源。"""
        ...

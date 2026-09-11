"""向量索引契约。

向量索引负责"语义相似"这一路召回。它与全文索引**互补而非替代**：
前者能召回换词表达的同一概念，后者能精确命中术语、缩写与数字，
两者通过 RRF 融合（SPEC §3.6）。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Protocol, runtime_checkable

from scitrace.domain import Fragment, SourceKey
from scitrace.ports.common import ScoredFragment

__all__ = ["VectorIndex", "VectorLike"]


#: 向量以 ``Sequence[float]`` 表达，而不是 numpy 数组——
#: 端口层不暴露实现细节，让"换成 FAISS / Qdrant / 向量数据库"不影响签名。
VectorLike = Sequence[float]


@runtime_checkable
class VectorIndex(Protocol):
    """片段向量的存储与相似检索。

    **归一化约定**：入库向量必须是 L2 归一化的（由 ``EmbeddingClient`` 保证），
    因此实现内部可以用点积代替余弦计算。实现应在 ``add`` 时校验这一前提，
    对未归一化的向量报错而不是默默给出错误的相似度。
    """

    @property
    def dimension(self) -> int:
        """向量维度。未加入任何数据时返回 0。"""
        ...

    def __len__(self) -> int:
        """索引中的片段数。"""
        ...

    async def add(self, items: Sequence[tuple[Fragment, VectorLike]]) -> None:
        """加入片段及其向量。

        对已存在的 ``fragment_id`` 应**覆盖**而非重复追加——
        重新摄入同一篇论文时不能留下陈旧副本。
        """
        ...

    async def search(
        self,
        query: VectorLike,
        k: int,
        *,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list[ScoredFragment]:
        """按余弦相似度返回 top-k。

        Args:
            query: 已归一化的查询向量。
            k: 返回条数。
            allowed_keys: 仅在此文献集合内检索。Agent 的
                ``search_literature`` 之后需要"在候选论文内取证"，
                这个参数就是为此存在——若没有它，取证阶段会越过用户/Agent
                选定的范围，产生不可解释的证据。

        Returns:
            按得分降序排列，``rank`` 字段从 1 开始填充。
        """
        ...

    async def mmr_search(
        self,
        query: VectorLike,
        k: int,
        *,
        fetch_k: int,
        lambda_: float = 0.5,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list[ScoredFragment]:
        """最大边际相关（MMR）检索：在相关性与结果多样性之间取平衡。

        实现定义（SPEC §3.6）::

            score(d) = λ · sim(q, d) − (1−λ) · max_{s ∈ 已选} sim(d, s)

        贪心选点。动机：论文里同一段论述常在相邻块中反复出现，
        纯 top-k 会把上下文塞满近乎重复的文本，浪费预算且降低覆盖面。

        Args:
            fetch_k: 候选池大小，必须 ≥ ``k``。
            lambda_: 0 = 只要多样性，1 = 退化为纯相似度。
        """
        ...

    async def remove(self, source_key: SourceKey) -> int:
        """删除某篇文献的全部片段。

        Returns:
            实际删除的片段数。文件被删除或更新时调用（SPEC §3.1 增量语义）。
        """
        ...

    def clear(self) -> None:
        """清空索引。"""
        ...

    def fragments(self) -> list[Fragment]:
        """返回索引中的全部片段（不含向量）。

        供 ``service`` 层做会话渲染与排障。**不含向量**是刻意的：
        调用方几乎总是只要文本，而返回向量会让这个方法的成本高一个量级。
        """
        ...

    async def persist(self) -> None:
        """把索引落盘。"""
        ...

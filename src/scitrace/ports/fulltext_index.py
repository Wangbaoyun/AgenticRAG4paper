"""全文索引契约。

全文索引承担"关键词精确召回"这一路：术语、缩写、基因名、数值、公式编号——
这些恰恰是稠密向量最容易糊掉的内容。中文场景下的切分方案见
``scitrace.util.tokenize_zh``（字符 bigram），本端口的实现应直接复用
``to_index_terms``，以保证**入库与查询两侧切分完全一致**。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable

from scitrace.domain import Fragment, SourceKey
from scitrace.ports.common import ScoredFragment

__all__ = ["FullTextIndex"]


@runtime_checkable
class FullTextIndex(Protocol):
    """基于关键词的片段检索。"""

    def __len__(self) -> int:
        """索引中的片段数。"""
        ...

    async def add(
        self,
        fragments: Sequence[Fragment],
        *,
        titles: Mapping[SourceKey, str] | None = None,
    ) -> None:
        """加入片段。

        Args:
            fragments: 待索引片段。
            titles: ``source_key -> 标题``。标题需要进索引以便
                按标题检索论文（Agent 的 ``search_literature`` 常用）。
                ``Fragment`` 本身不含标题，因此必须显式传入。

        Note:
            多数全文索引后端的写入是**缓冲**的，``add`` 之后必须调用
            :meth:`commit` 才对 :meth:`search` 可见。契约刻意把这两步分开，
            使批量摄入不必每篇提交一次。
        """
        ...

    async def search(
        self,
        query: str,
        k: int,
        *,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list[ScoredFragment]:
        """按 BM25 类得分返回 top-k。

        Args:
            query: 原始查询串（**未**预处理）。实现负责按索引时的同一规则切分，
                这是最容易出错的地方：两侧切分不一致会导致静默的零召回。
            k: 返回条数。
            allowed_keys: 仅在此文献集合内检索。

        Returns:
            按得分降序排列，``rank`` 从 1 开始填充。

        Note:
            得分的量纲由后端决定且**无上界**，因此只能用于本路内排序；
            跨路融合必须走基于名次的 RRF（见 ``pipeline.retrieval``）。
        """
        ...

    async def remove(self, source_key: SourceKey) -> int:
        """删除某篇文献的全部片段，返回删除数。"""
        ...

    async def commit(self) -> None:
        """提交挂起的写入。"""
        ...

    def clear(self) -> None:
        """清空索引。"""
        ...

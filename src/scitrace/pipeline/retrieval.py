"""检索内核：四种策略，可切换、可对照、可消融。【原创增量 ②】

## 为什么把"融合"做成基于名次而不是得分

向量相似度是余弦（有界、含义明确），BM25 是无上界的对数似然，交叉编码器输出的是
logits。三者的数值范围与分布毫无可比性：把它们加权相加，权重就变成了一个
**随数据集漂移的魔数**——换个语料就要重调，而调参过程无法解释。

Reciprocal Rank Fusion 只用**名次**：``score(d) = Σ_r 1/(rrf_k + rank_r(d))``。
名次是无量纲的，因此不需要归一化、不需要调权重，且对任一路的得分尺度完全不敏感。
代价是丢掉了"第一名领先第二名多少"的信息——在候选池本来就要交给下游重排的前提下，
这个信息本来也用不上。

## 四种策略的定位

| 策略 | 用途 |
| --- | --- |
| ``dense`` | 基线。只看向量召回，用于回答"混合检索到底有没有用" |
| ``dense_mmr`` | 默认。在向量召回上加多样性约束，缓解相邻块内容重复 |
| ``hybrid_rrf`` | 加入关键词精确召回，改善术语、缩写、数字的命中 |
| ``hybrid_rrf_rerank`` | 在混合召回后接交叉编码器重排 |

四者共用同一套返回类型，因此消融实验只需改一个配置项。
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence

from scitrace.config import RetrievalSettings
from scitrace.domain import Fragment, SourceKey
from scitrace.ports import EmbeddingClient, FullTextIndex, ScoredFragment, VectorIndex
from scitrace.ports.reranker import Reranker

logger = logging.getLogger(__name__)

__all__ = ["Retriever", "reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredFragment]],
    *,
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[ScoredFragment]:
    """把多路检索结果按 RRF 融合成一路。

    ``score(d) = Σ_r 1 / (rrf_k + rank_r(d))``，其中 ``rank`` 从 1 开始。

    Args:
        result_lists: 各路结果，每路已按各自的相关性降序排列。
        rrf_k: 平滑常数。取值越大，名次差异被压得越平（第一名的优势越小）。
            60 是文献中的常用默认值：它让第 1 名与第 2 名的得分比约为
            ``(60+2)/(60+1) ≈ 1.016``，即名次只带来温和的优势，
            避免任何单路的结果垄断最终列表。
        top_n: 返回条数上限；``None`` 表示全部返回。

    Returns:
        融合后的结果，按 RRF 得分降序。同一 ``fragment_id`` 在多路中出现时会累加得分
        （这正是 RRF 奖励"被多路同时召回"的方式），并保留**首次出现**时的片段对象。

    Note:
        名次取 ``ScoredFragment.rank``；若某路没有填充 ``rank``（为 0），
        则退化为按该列表的顺序位次（1-based）。这样即使某个后端忘了填 ``rank``，
        融合结果依然有意义，而不是把所有条目算成同一个名次。
    """
    if rrf_k < 1:
        raise ValueError(f"rrf_k 必须 >= 1，得到 {rrf_k}")

    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredFragment] = {}
    for results in result_lists:
        for position, item in enumerate(results, start=1):
            rank = item.rank if item.rank > 0 else position
            fragment_id = item.fragment.fragment_id
            scores[fragment_id] = scores.get(fragment_id, 0.0) + 1.0 / (rrf_k + rank)
            first_seen.setdefault(fragment_id, item)

    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_n is not None:
        ordered = ordered[: max(0, top_n)]
    return [
        ScoredFragment(
            fragment=first_seen[fragment_id].fragment,
            score=score,
            origin="rrf",
            rank=position,
        )
        for position, (fragment_id, score) in enumerate(ordered, start=1)
    ]


class Retriever:
    """按配置策略执行检索。

    本类**不做筛选、不做摘要**——那些属于 ``pipeline.screening``。
    保持这个边界让"检索策略"与"证据加工方式"可以独立消融：
    仅改变 :class:`~scitrace.config.RetrievalSettings` 就能切换四路策略之一，
    而筛选器完全不必知道检索是怎么做的。
    """

    def __init__(
        self,
        *,
        vector_index: VectorIndex | None,
        fulltext_index: FullTextIndex | None,
        embedder: EmbeddingClient | None,
        settings: RetrievalSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.fulltext_index = fulltext_index
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        allowed_keys: Collection[SourceKey] | None = None,
    ) -> list[ScoredFragment]:
        """按配置策略检索。

        Args:
            query: 查询串。
            k: 返回条数；``None`` 时取配置的 ``retrieval.k``。
            allowed_keys: 仅在该文献集合内检索（Agent 选定候选论文后使用）。

        Returns:
            按相关性降序排列的结果。索引为空或无命中时返回空列表。

        Raises:
            ValueError: 配置的策略需要某个未提供的部件。
                刻意**不静默降级**——见下。
        """
        top_k = self.settings.k if k is None else k
        if top_k <= 0 or not query.strip():
            return []

        strategy = self.settings.strategy
        if strategy == "dense":
            return await self._dense(query, top_k, allowed_keys)
        if strategy == "dense_mmr":
            return await self._dense_mmr(query, top_k, allowed_keys)
        if strategy == "hybrid_rrf":
            return await self._hybrid_rrf(query, top_k, allowed_keys)
        if strategy == "hybrid_rrf_rerank":
            return await self._hybrid_rrf_rerank(query, top_k, allowed_keys)
        raise ValueError(f"未知的检索策略 {strategy!r}")

    # ---------------------------------------------------------------- 各路实现 --

    async def _query_vector(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        vectors = await self.embedder.embed([query], kind="query")
        return vectors[0] if vectors else None

    async def _dense(
        self, query: str, k: int, allowed_keys: Collection[SourceKey] | None
    ) -> list[ScoredFragment]:
        self._require(self.vector_index, self.embedder, strategy="dense")
        vector = await self._query_vector(query)
        if vector is None:  # pragma: no cover - 由 _require 保证
            return []
        return await self.vector_index.search(vector, k, allowed_keys=allowed_keys)  # type: ignore[union-attr]

    async def _dense_mmr(
        self, query: str, k: int, allowed_keys: Collection[SourceKey] | None
    ) -> list[ScoredFragment]:
        self._require(self.vector_index, self.embedder, strategy="dense_mmr")
        vector = await self._query_vector(query)
        if vector is None:  # pragma: no cover
            return []
        fetch_k = max(k, k * self.settings.fetch_k_multiplier)
        return await self.vector_index.mmr_search(  # type: ignore[union-attr]
            vector,
            k,
            fetch_k=fetch_k,
            lambda_=self.settings.mmr_lambda,
            allowed_keys=allowed_keys,
        )

    async def _hybrid_rrf(
        self,
        query: str,
        k: int,
        allowed_keys: Collection[SourceKey] | None,
        *,
        pool: int | None = None,
    ) -> list[ScoredFragment]:
        """向量路与关键词路各自召回，再按 RRF 融合。

        ``pool`` 控制每路的候选数量。默认与 ``k`` 相同并不合理——
        融合的价值恰恰来自"两路召回的东西不完全一样"，
        过小的候选池会让两路高度重叠，融合退化为单路。
        因此取 ``k * rerank_pool_multiplier``（默认 3 倍）。
        """
        self._require(self.vector_index, self.embedder, strategy="hybrid_rrf")
        per_route = pool if pool is not None else max(k, k * self.settings.rerank_pool_multiplier)

        vector = await self._query_vector(query)
        dense_results = (
            await self.vector_index.search(vector, per_route, allowed_keys=allowed_keys)  # type: ignore[union-attr]
            if vector is not None
            else []
        )

        bm25_results: list[ScoredFragment] = []
        if self.fulltext_index is not None:
            bm25_results = await self.fulltext_index.search(
                query, per_route, allowed_keys=allowed_keys
            )
        else:
            logger.debug("未配置全文索引，混合检索退化为纯向量检索")
            # 向量索引已在 _require 中确认可用

        if not bm25_results:
            # 没有任何关键词结果时，RRF 与纯向量排序等价，但得分会被重新标定为 RRF 量纲。
            # 这会让"混合检索"的消融对照混入一个量纲变化，因此直接返回原结果。
            return dense_results[:k]

        return reciprocal_rank_fusion(
            [dense_results, bm25_results], rrf_k=self.settings.rrf_k, top_n=k
        )

    async def _hybrid_rrf_rerank(
        self, query: str, k: int, allowed_keys: Collection[SourceKey] | None
    ) -> list[ScoredFragment]:
        if self.reranker is None:
            # SPEC §3.6：必须显式报错，不得静默降级。
            # 静默降级会让"开了重排"的实验数据其实是没重排的，
            # 而这种错误在结果表里完全看不出来。
            raise ValueError(
                f"检索策略 {self.settings.strategy!r} 需要重排器，但未提供。"
                "请安装重排依赖（pip install scitrace[rerank]），"
                "或把 retrieval.strategy 改为 hybrid_rrf。"
            )
        pool = max(k, k * self.settings.rerank_pool_multiplier)
        candidates = await self._hybrid_rrf(query, pool, allowed_keys, pool=pool)
        if not candidates:
            return []
        fragments: Sequence[Fragment] = [item.fragment for item in candidates]
        return await self.reranker.rerank(query, fragments, top_n=k)

    @staticmethod
    def _require(*parts: object, strategy: str) -> None:
        """校验策略所需的部件都已提供。"""
        missing = [
            name
            for name, part in zip(("向量索引", "嵌入器"), parts, strict=True)
            if part is None
        ]
        if missing:
            raise ValueError(
                f"检索策略 {strategy!r} 需要 {'、'.join(missing)}，但未提供。"
                "请在装配层（scitrace.factory）补齐，或改用不需要它们的策略。"
            )

    async def aclose(self) -> None:
        """释放重排器资源。索引与嵌入器由装配层负责关闭。"""
        if self.reranker is not None:
            await self.reranker.aclose()

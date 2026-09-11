"""测试检索内核与 RRF 融合。

``reciprocal_rank_fusion`` 是纯函数，因此可以被穷尽测试——这很重要，
因为"融合"是四路检索策略里唯一有算法判断的部分，错了不会报错，
只会让检索结果悄悄变差。
"""

from __future__ import annotations

import pytest
from fakes import FakeEmbedder, FakeFullTextIndex, FakeVectorIndex
from scitrace.config import RetrievalSettings
from scitrace.domain import Fragment
from scitrace.pipeline.retrieval import Retriever, reciprocal_rank_fusion
from scitrace.ports import ScoredFragment

DIM = 8


def make_fragment(fragment_id: str, text: str, source_key: str = "src-1") -> Fragment:
    return Fragment(
        fragment_id=fragment_id,
        source_key=source_key,
        text=text,
        chunk_index=0,
        char_count=len(text),
    )


def scored(fragment_id: str, score: float, rank: int = 0, origin: str = "dense") -> ScoredFragment:
    return ScoredFragment(
        fragment=make_fragment(fragment_id, f"text of {fragment_id}"),
        score=score,
        origin=origin,
        rank=rank,
    )


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self) -> None:
        fused = reciprocal_rank_fusion([[scored("a", 1.0), scored("b", 0.5)]])
        assert [item.fragment.fragment_id for item in fused] == ["a", "b"]

    def test_ranks_are_assigned_from_one(self) -> None:
        fused = reciprocal_rank_fusion([[scored("a", 1.0), scored("b", 0.5), scored("c", 0.1)]])
        assert [item.rank for item in fused] == [1, 2, 3]
        assert all(item.origin == "rrf" for item in fused)

    def test_document_in_both_lists_wins(self) -> None:
        """RRF 奖励"被多路同时召回"——这正是融合能提升召回的理由。"""
        dense = [scored("a", 0.9), scored("b", 0.8)]
        bm25 = [scored("b", 12.0), scored("c", 9.0)]
        fused = reciprocal_rank_fusion([dense, bm25])
        assert fused[0].fragment.fragment_id == "b"

    def test_score_magnitudes_are_ignored(self) -> None:
        """BM25 得分无上界、余弦有界，直接相加会让权重变成随数据集漂移的魔数。

        RRF 只看名次，因此把一个列表的得分整体放大一千倍，结果不变。
        """
        dense = [scored("a", 0.9), scored("b", 0.8)]
        bm25_small = [scored("b", 1.0), scored("c", 0.5)]
        bm25_huge = [scored("b", 1000.0), scored("c", 500.0)]
        assert [f.fragment.fragment_id for f in reciprocal_rank_fusion([dense, bm25_small])] == [
            f.fragment.fragment_id for f in reciprocal_rank_fusion([dense, bm25_huge])
        ]

    def test_falls_back_to_position_when_rank_missing(self) -> None:
        """某个后端忘了填 rank 时，融合结果仍应有意义，

        而不是把所有条目算成同一个名次（那会让顺序完全由字典序决定）。
        """
        fused = reciprocal_rank_fusion([[scored("a", 1.0), scored("b", 0.9)]])
        assert fused[0].fragment.fragment_id == "a"

    def test_top_n_truncates(self) -> None:
        fused = reciprocal_rank_fusion([[scored(str(i), 1.0) for i in range(10)]], top_n=3)
        assert len(fused) == 3

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="rrf_k"):
            reciprocal_rank_fusion([[scored("a", 1.0)]], rrf_k=0)

    def test_larger_rrf_k_flattens_rank_advantage(self) -> None:
        """rrf_k 越大，**同一路内**相邻名次的得分差越小——这是它唯一的语义。

        注意不要把"多路命中加成"混进来：一个被两路同时召回的文档，
        其相对优势反而会随 rrf_k 增大而变大，方向与本测试相反。
        这里只比较单路内的第 1 名与第 2 名。
        """
        single = [[scored("a", 1.0), scored("b", 0.9)]]
        flat = reciprocal_rank_fusion(single, rrf_k=1000)
        sharp = reciprocal_rank_fusion(single, rrf_k=1)
        assert flat[0].score / flat[1].score < sharp[0].score / sharp[1].score
        assert flat[0].score / flat[1].score < 1.01

    def test_result_is_deterministic(self) -> None:
        lists = [[scored("a", 1.0), scored("b", 1.0)]]
        assert [f.fragment.fragment_id for f in reciprocal_rank_fusion(lists)] == [
            f.fragment.fragment_id for f in reciprocal_rank_fusion(lists)
        ]


class Harness:
    def __init__(self) -> None:
        self.vector = FakeVectorIndex()
        self.fulltext = FakeFullTextIndex()
        self.embedder = FakeEmbedder(dimension=DIM)

    async def seed(self, fragments: list[Fragment]) -> None:
        vectors = await self.embedder.embed([item.text for item in fragments])
        await self.vector.add(list(zip(fragments, vectors, strict=True)))
        await self.fulltext.add(fragments)
        await self.fulltext.commit()

    def retriever(self, **overrides) -> Retriever:
        settings = RetrievalSettings(**overrides)
        return Retriever(
            vector_index=self.vector,
            fulltext_index=self.fulltext,
            embedder=self.embedder,
            settings=settings,
        )


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    @property
    def model_name(self) -> str:
        return "fake-reranker"

    async def rerank(self, query: str, fragments, *, top_n: int):
        self.calls += 1
        return [
            ScoredFragment(fragment=item, score=1.0 / (index + 1), origin="rerank", rank=index + 1)
            for index, item in enumerate(fragments[:top_n])
        ]

    async def aclose(self) -> None:
        self.closed = True


class TestStrategies:
    async def test_dense_returns_results(self) -> None:
        harness = Harness()
        await harness.seed([make_fragment("f1", "证据可溯源问答系统的检索方法")])
        results = await harness.retriever(strategy="dense").retrieve("证据可溯源", k=5)
        assert results
        assert results[0].origin == "dense"

    async def test_dense_mmr_returns_ranked_results(self) -> None:
        harness = Harness()
        await harness.seed([make_fragment(f"f{i}", f"关于检索的第{i}段中文内容") for i in range(5)])
        results = await harness.retriever(strategy="dense_mmr").retrieve("检索", k=3)
        assert len(results) <= 3
        assert [item.rank for item in results] == list(range(1, len(results) + 1))

    async def test_hybrid_rrf_uses_both_routes(self) -> None:
        harness = Harness()
        await harness.seed(
            [
                make_fragment("f1", "证据可溯源问答系统的检索方法"),
                make_fragment("f2", "完全无关的另一段中文内容讲述气候模型"),
            ]
        )
        results = await harness.retriever(strategy="hybrid_rrf").retrieve("证据可溯源", k=5)
        assert results
        assert results[0].origin == "rrf"

    async def test_empty_index_returns_empty(self) -> None:
        harness = Harness()
        assert await harness.retriever(strategy="dense").retrieve("anything", k=5) == []

    @pytest.mark.parametrize("query", ["", "   "])
    async def test_blank_query_returns_empty(self, query: str) -> None:
        harness = Harness()
        await harness.seed([make_fragment("f1", "内容")])
        assert await harness.retriever().retrieve(query, k=5) == []

    async def test_k_zero_returns_empty(self) -> None:
        harness = Harness()
        await harness.seed([make_fragment("f1", "内容")])
        assert await harness.retriever().retrieve("内容", k=0) == []

    async def test_allowed_keys_scopes_results(self) -> None:
        harness = Harness()
        await harness.seed(
            [
                make_fragment("f1", "证据抽取方法", source_key="src-a"),
                make_fragment("f2", "证据抽取方法", source_key="src-b"),
            ]
        )
        results = await harness.retriever(strategy="dense").retrieve(
            "证据抽取", k=10, allowed_keys={"src-a"}
        )
        assert results
        assert {item.fragment.source_key for item in results} == {"src-a"}

    async def test_rerank_strategy_requires_reranker(self) -> None:
        """SPEC §3.6：必须显式报错，不得静默降级。

        静默降级会让"开了重排"的实验数据其实是没重排的，而结果表里完全看不出来。
        """
        harness = Harness()
        await harness.seed([make_fragment("f1", "内容")])
        with pytest.raises(ValueError, match="需要重排器"):
            await harness.retriever(strategy="hybrid_rrf_rerank").retrieve("内容", k=5)

    async def test_rerank_strategy_uses_reranker(self) -> None:
        harness = Harness()
        await harness.seed([make_fragment(f"f{i}", f"第{i}段关于检索的内容") for i in range(4)])
        reranker = FakeReranker()
        retriever = Retriever(
            vector_index=harness.vector,
            fulltext_index=harness.fulltext,
            embedder=harness.embedder,
            settings=RetrievalSettings(strategy="hybrid_rrf_rerank"),
            reranker=reranker,
        )
        results = await retriever.retrieve("检索", k=2)
        assert reranker.calls == 1
        assert results and results[0].origin == "rerank"
        assert len(results) == 2

    async def test_reranker_is_closed(self) -> None:
        reranker = FakeReranker()
        retriever = Retriever(
            vector_index=FakeVectorIndex(),
            fulltext_index=FakeFullTextIndex(),
            embedder=FakeEmbedder(dimension=DIM),
            settings=RetrievalSettings(),
            reranker=reranker,
        )
        await retriever.aclose()
        assert reranker.closed is True

    @pytest.mark.parametrize("strategy", ["dense", "dense_mmr"])
    async def test_missing_embedder_raises(self, strategy: str) -> None:
        retriever = Retriever(
            vector_index=FakeVectorIndex(),
            fulltext_index=FakeFullTextIndex(),
            embedder=None,
            settings=RetrievalSettings(strategy=strategy),
        )
        with pytest.raises(ValueError, match="嵌入器"):
            await retriever.retrieve("q", k=3)

    async def test_unknown_strategy_raises(self) -> None:
        retriever = Harness().retriever()
        object.__setattr__(retriever.settings, "strategy", "magic")
        with pytest.raises(ValueError, match="未知的检索策略"):
            await retriever.retrieve("q", k=3)

    async def test_hybrid_degrades_to_dense_without_fulltext(self) -> None:
        """没有全文索引时混合检索退化为纯向量——这是配置缺失，不是错误。

        与"需要重排器却没有"不同：后者会改变实验结论，前者只是少了一路召回。
        """
        harness = Harness()
        await harness.seed([make_fragment("f1", "关于检索的中文内容")])
        retriever = Retriever(
            vector_index=harness.vector,
            fulltext_index=None,
            embedder=harness.embedder,
            settings=RetrievalSettings(strategy="hybrid_rrf"),
        )
        results = await retriever.retrieve("检索", k=3)
        assert results
        assert results[0].origin == "dense"

    async def test_hybrid_pool_is_larger_than_k(self) -> None:
        """融合的价值来自"两路召回的东西不完全一样"，

        过小的候选池会让两路高度重叠、融合退化为单路。
        """
        harness = Harness()
        await harness.seed([make_fragment(f"f{i}", f"第{i}段关于检索与召回的内容") for i in range(12)])
        results = await harness.retriever(
            strategy="hybrid_rrf", k=2, rerank_pool_multiplier=3
        ).retrieve("检索", k=2)
        assert len(results) == 2

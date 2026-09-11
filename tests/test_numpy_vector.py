"""测试 numpy 向量索引适配器。

覆盖 SPEC §3.4（向量索引）与 §3.6（MMR）规定的外部可观察行为，重点钉住四类
"错了也不会立刻报错"的语义：

1. **归一化前提**：未归一化/零向量必须在 ``add`` 处失败，而不是让余弦检索悄悄算错；
2. **先过滤再取 top-k**：``allowed_keys`` 场景下返回条数不能因为过滤而变少；
3. **MMR 真的去冗余**：用一个"5 个近重复 + 1 个离群"的数据集，断言 MMR 结果的
   两两相似度显著低于纯向量检索，而不只是断言"能跑"；
4. **持久化往返**：``persist`` → ``load`` 之后检索结果（id 与得分）必须一致。

测试数据全部为**虚构**的作者名、标题与 DOI（``10.5555/…`` 示例前缀），
不使用任何真实论文的数据。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from scitrace.adapters.indexes.numpy_vector import NumpyVectorIndex, _top_k_order
from scitrace.domain import Fragment
from scitrace.ports import ScoredFragment, VectorIndex

#: 向量维度。取 16 维是为了让"近重复簇"与"离群点"能各自占用正交方向。
DIM = 16


# ---- 虚构测试数据 ---------------------------------------------------------


def make_fragment(fragment_id: str, source_key: str, *, chunk_index: int = 0) -> Fragment:
    """构造一个虚构片段。"""
    text = f"Fictional passage {fragment_id} about adaptive evidence retrieval."
    return Fragment(
        fragment_id=fragment_id,
        source_key=source_key,
        text=text,
        chunk_index=chunk_index,
        section_path=["3 Method"],
        page_start=1,
        page_end=2,
        char_count=len(text),
    )


def normalize(values: Sequence[float]) -> np.ndarray:
    """L2 归一化为 float32 向量——模拟 EmbeddingClient 的输出。"""
    array = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    assert norm > 0.0, "测试数据本身不应含零向量"
    return (array / norm).astype(np.float32)


def basis(index: int, dim: int = DIM) -> np.ndarray:
    """第 ``index`` 个标准基向量。"""
    values = np.zeros(dim, dtype=np.float64)
    values[index] = 1.0
    return values


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """两个（已归一化或未归一化的）向量的余弦相似度，仅用于测试断言。"""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def vector_with_cosine(cosine_value: float, axis: int) -> np.ndarray:
    """与查询（``basis(0)``）余弦恰为 ``cosine_value`` 的单位向量。

    把与查询正交的那个分量放到 ``axis`` 轴上，于是不同 ``axis`` 的向量即使余弦相同
    也彼此不同，且相似度是解析可写出的确定值。
    """
    assert 0 < axis < DIM, "扰动轴不能是查询轴本身"
    tail = float(np.sqrt(max(0.0, 1.0 - cosine_value**2)))
    return normalize(cosine_value * basis(0) + tail * basis(axis))


def max_pairwise_cosine(vectors: Sequence[Sequence[float]]) -> float:
    """一组向量两两余弦相似度的最大值（0 个或 1 个向量时返回 0.0）。"""
    if len(vectors) < 2:
        return 0.0
    matrix = np.asarray(vectors, dtype=np.float64)
    similarity = matrix @ matrix.T / np.outer(
        np.linalg.norm(matrix, axis=1), np.linalg.norm(matrix, axis=1)
    )
    return float(similarity[np.triu_indices(len(vectors), k=1)].max())


class Corpus:
    """一个小型虚构语料：三个 source_key，各带若干片段与已知相似度。

    向量的构造方式是"查询轴 + 若干正交扰动轴"，因此每个片段与查询的余弦相似度
    是已知的解析值，断言里可以直接写数字而不是同样由 numpy 算出来的期望值。
    """

    #: 虚构文献的稳定键。
    KEYS = ("aaaa000000000001", "bbbb000000000002", "cccc000000000003")

    def __init__(self) -> None:
        self.fragments: list[Fragment] = []
        self.vectors: list[np.ndarray] = []

        # s1 的三个片段与查询的余弦相似度依次为 0.90 / 0.80 / 0.70（全局排名靠后）。
        for rank, cosine_value in enumerate((0.90, 0.80, 0.70)):
            self._add(f"s1-{rank}", self.KEYS[0], cosine_value)
        # s2 的三个片段相似度更高（0.99 / 0.98 / 0.97），全局 top-2 全部来自 s2。
        for rank, cosine_value in enumerate((0.99, 0.98, 0.97)):
            self._add(f"s2-{rank}", self.KEYS[1], cosine_value)
        # s3 是"完全不相关"的文献，与查询正交。
        for rank in range(3):
            self._add(f"s3-{rank}", self.KEYS[2], 0.0)

    def _add(self, fragment_id: str, source_key: str, cosine_value: float) -> None:
        """加入一个与查询相似度恰为 ``cosine_value`` 的向量（扰动方向逐条错开）。"""
        axis = len(self.fragments) % (DIM - 1)
        if axis == 0:
            # 0 号轴是查询轴本身，不能用作扰动方向，否则相似度不再是解析值。
            axis = DIM - 1
        vector = vector_with_cosine(cosine_value, axis)
        self.fragments.append(
            make_fragment(fragment_id, source_key, chunk_index=len(self.fragments))
        )
        self.vectors.append(vector)

    def items(self) -> list[tuple[Fragment, np.ndarray]]:
        return list(zip(self.fragments, self.vectors, strict=True))

    def vector_of(self, fragment_id: str) -> np.ndarray:
        for fragment, vector in zip(self.fragments, self.vectors, strict=True):
            if fragment.fragment_id == fragment_id:
                return vector
        raise KeyError(fragment_id)


def query_vector() -> np.ndarray:
    """查询向量 = 第一个基向量。"""
    return normalize(basis(0))


def mmr_cluster_items() -> tuple[np.ndarray, list[tuple[Fragment, np.ndarray]], str]:
    """构造 MMR 测试集：5 个近重复片段 + 1 个"相关性略低但方向完全不同"的片段。

    返回 ``(query, items, 离群片段 id)``。几何设计：近重复簇围绕与查询余弦约 0.70 的
    方向 ``c`` 紧密排布（相互余弦约 0.9996），离群片段与查询余弦 0.65、与簇内任何
    成员余弦仅约 0.455。因此纯 top-k 会把簇内成员全数召回，而 MMR 应当在第 2 位
    就换成离群片段。
    """
    cluster_direction = normalize(0.70 * basis(0) + 0.714 * basis(1))
    items: list[tuple[Fragment, np.ndarray]] = []
    for index in range(5):
        vector = normalize(cluster_direction + (0.020 + 0.002 * index) * basis(2 + index))
        items.append((make_fragment(f"cluster-{index}", Corpus.KEYS[0], chunk_index=index), vector))
    outlier = normalize(0.65 * basis(0) + 0.76 * basis(9))
    items.append((make_fragment("outlier", Corpus.KEYS[1], chunk_index=5), outlier))
    return normalize(basis(0)), items, "outlier"


# ---- 基本契约 -------------------------------------------------------------


class TestConstructionAndProtocol:
    def test_starts_empty_with_unknown_dimension(self) -> None:
        index = NumpyVectorIndex()
        assert len(index) == 0
        assert index.dimension == 0
        assert index.fragments() == []
        assert index.path is None

    def test_explicit_dimension_is_reported_before_any_data(self) -> None:
        """显式给出维度时空索引也报该维度（配置已知，不必等数据）。"""
        assert NumpyVectorIndex(dimension=8).dimension == 8

    def test_negative_dimension_rejected(self) -> None:
        with pytest.raises(ValueError):
            NumpyVectorIndex(dimension=-1)

    def test_satisfies_vector_index_protocol(self) -> None:
        """实现必须满足 ``VectorIndex`` 端口（方法名与属性一个都不能少）。"""
        assert isinstance(NumpyVectorIndex(), VectorIndex)


class TestAdd:
    async def test_len_and_dimension_autodetected(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        assert len(index) == len(corpus.fragments)
        assert index.dimension == DIM

    async def test_empty_sequence_is_noop(self) -> None:
        index = NumpyVectorIndex()
        await index.add([])
        assert len(index) == 0
        assert index.dimension == 0

    async def test_duplicate_fragment_id_overwrites_in_place(self) -> None:
        """重复 add 同一 fragment_id 必须原地覆盖，不能追加出重复行。"""
        index = NumpyVectorIndex()
        fragment = make_fragment("dup", Corpus.KEYS[0])
        await index.add([(fragment, normalize(basis(0)))])
        await index.add([(fragment, normalize(basis(1)))])

        assert len(index) == 1
        assert index.fragments() == [fragment]
        # 覆盖后应当用新向量检索得到：查询 e0 时旧向量得分为 1，新向量为 0。
        hits = await index.search(normalize(basis(0)), 1)
        assert hits[0].fragment.fragment_id == "dup"
        assert hits[0].score == pytest.approx(0.0, abs=1e-6)

    async def test_overwrite_replaces_the_stored_fragment(self) -> None:
        index = NumpyVectorIndex()
        await index.add([(make_fragment("dup", Corpus.KEYS[0]), normalize(basis(0)))])
        replacement = make_fragment("dup", Corpus.KEYS[1])
        await index.add([(replacement, normalize(basis(2)))])

        assert len(index) == 1
        assert index.fragments() == [replacement]

    async def test_batch_with_invalid_item_leaves_index_untouched(self) -> None:
        """整批校验通过才写入：第 2 条非法时第 1 条也不能进库。"""
        index = NumpyVectorIndex()
        good = make_fragment("good", Corpus.KEYS[0])
        bad = make_fragment("bad", Corpus.KEYS[0])
        with pytest.raises(ValueError):
            await index.add([(good, normalize(basis(0))), (bad, 3.0 * basis(1))])

        assert len(index) == 0
        assert index.dimension == 0

    @pytest.mark.parametrize("scale", [0.5, 2.0, 1.01])
    async def test_unnormalized_vector_rejected(self, scale: float) -> None:
        index = NumpyVectorIndex()
        fragment = make_fragment("not-unit", Corpus.KEYS[0])
        with pytest.raises(ValueError) as excinfo:
            await index.add([(fragment, scale * basis(0))])

        message = str(excinfo.value)
        assert "not-unit" in message, "错误消息必须包含 fragment_id 以便定位"
        assert f"{scale:.6f}" in message, "错误消息必须包含实际范数"

    async def test_zero_vector_rejected_separately(self) -> None:
        index = NumpyVectorIndex()
        fragment = make_fragment("zero", Corpus.KEYS[0])
        with pytest.raises(ValueError, match="零向量"):
            await index.add([(fragment, np.zeros(DIM, dtype=np.float32))])

    async def test_empty_vector_rejected(self) -> None:
        index = NumpyVectorIndex()
        with pytest.raises(ValueError, match="零向量"):
            await index.add([(make_fragment("empty", Corpus.KEYS[0]), [])])

    @pytest.mark.parametrize(
        "shape_error",
        [
            pytest.param(normalize(basis(0)).reshape(1, -1), id="行切片-2维"),
            pytest.param(np.float32(1.0), id="标量-0维"),
        ],
    )
    async def test_non_1d_vector_rejected(self, shape_error: np.ndarray) -> None:
        """非 1 维输入必须被拒。

        典型来源是调用方误把"一行切片" ``vectors[i:i+1]`` 而不是 ``vectors[i]`` 传进来：
        范数校验会照常通过，而维度校验在**空索引**上（此时维度尚未确定）根本不会触发，
        最后索引里会出现一个形状错误的矩阵——所以这条单独钉住形状本身。
        """
        index = NumpyVectorIndex()
        with pytest.raises(ValueError, match="1 维"):
            await index.add([(make_fragment("odd-shape", Corpus.KEYS[0]), shape_error)])
        assert len(index) == 0

    async def test_non_finite_vector_rejected(self) -> None:
        """NaN 会绕过 `abs(norm - 1) > tol` 这类比较，必须单独拦下。"""
        index = NumpyVectorIndex()
        nan_vector = normalize(basis(0))
        nan_vector[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            await index.add([(make_fragment("nan", Corpus.KEYS[0]), nan_vector)])

    async def test_dimension_mismatch_with_existing_index_rejected(self) -> None:
        index = NumpyVectorIndex()
        await index.add([(make_fragment("first", Corpus.KEYS[0]), normalize(basis(0)))])

        with pytest.raises(ValueError) as excinfo:
            await index.add([(make_fragment("wrong", Corpus.KEYS[0]), normalize(basis(0, dim=4)))])

        message = str(excinfo.value)
        assert "4" in message and str(DIM) in message, "消息要同时给出实际维度与期望维度"
        assert len(index) == 1

    async def test_dimension_mismatch_within_one_batch_rejected(self) -> None:
        index = NumpyVectorIndex()
        with pytest.raises(ValueError):
            await index.add(
                [
                    (make_fragment("a", Corpus.KEYS[0]), normalize(basis(0))),
                    (make_fragment("b", Corpus.KEYS[0]), normalize(basis(0, dim=4))),
                ]
            )
        assert len(index) == 0

    async def test_explicit_dimension_is_enforced_on_first_add(self) -> None:
        """构造时给了维度，首次 add 就必须按它校验（不能因为索引还空着就放行）。"""
        index = NumpyVectorIndex(dimension=4)

        with pytest.raises(ValueError):
            await index.add([(make_fragment("wider", Corpus.KEYS[0]), normalize(basis(0)))])

        await index.add([(make_fragment("ok", Corpus.KEYS[0]), normalize(basis(0, dim=4)))])
        assert len(index) == 1
        assert index.dimension == 4

    async def test_concurrent_calls_do_not_leak_intermediate_state(self) -> None:
        """并发 ``await`` 必须安全：每个方法体内部没有 ``await``，因此不会中途让出。

        用 ``gather`` 同时发起多批入库，要求全部生效且长度精确——若实现里存在
        "写一半就 ``await`` 一下"的写法，这里会稳定地丢更新或让行号映射错位。
        """
        index = NumpyVectorIndex()
        batches = [
            [
                (
                    make_fragment(f"batch{group}-{rank}", Corpus.KEYS[group], chunk_index=rank),
                    vector_with_cosine(0.9 - 0.1 * rank, axis=group + 1),
                )
                for rank in range(3)
            ]
            for group in range(3)
        ]

        await asyncio.gather(*(index.add(batch) for batch in batches))

        assert len(index) == 9
        assert index.dimension == DIM
        assert len({fragment.fragment_id for fragment in index.fragments()}) == 9

        dense, diverse = await asyncio.gather(
            index.search(query_vector(), 3),
            index.mmr_search(query_vector(), 3, fetch_k=9),
        )
        assert [hit.fragment.fragment_id for hit in dense] == ["batch0-0", "batch1-0", "batch2-0"]
        assert len(diverse) == 3


class TestTopKSelection:
    """直接测试 top-k 选择这一纯函数。

    为什么绕过公开方法测它：候选数 > k 时实现走 ``argpartition``（只保证选出最大的 k 个，
    **不保证顺序**），随后必须补一次排序。用公开 API 加小规模语料无法稳定地观察到
    "漏掉那次排序"——numpy 的 introselect 在小数组上常常恰好返回有序的 k 个。
    排序契约本身与 numpy 版本无关，因此在这里用显式断言钉住它。
    """

    def test_small_input_is_sorted_descending(self) -> None:
        scores = np.array([0.50, 0.95, 0.60, 0.99, 0.95, 0.90], dtype=np.float32)

        assert _top_k_order(scores, 6).tolist() == [3, 1, 4, 5, 2, 0]
        assert _top_k_order(scores, 3).tolist() == [3, 1, 4]
        assert _top_k_order(scores, 1).tolist() == [3]

    def test_large_input_matches_full_stable_sort(self) -> None:
        """候选远多于 k 时，返回的顺序必须与"全量稳定排序后取前 k"完全一致。

        固定用 seed=5 生成这组得分是刻意的：在这组数据上，底层选择过程返回的 k 个
        位置**本身就是乱序的**（它只保证"选出最大的 k 个"），因此这条断言真正压到了
        选择之后的排序步骤，而不是碰巧通过。断言本身只依赖契约（顺序 = 稳定降序），
        与 numpy 的内部实现无关，换版本也不会误报。
        """
        rng = np.random.default_rng(5)
        scores = rng.random(300).astype(np.float32)
        expected = np.lexsort((np.arange(300), -scores))[:150]

        order = _top_k_order(scores, 150)

        assert order.tolist() == expected.tolist()
        assert scores[order].tolist() == sorted(scores.tolist(), reverse=True)[:150]

    def test_ties_break_by_insertion_order(self) -> None:
        scores = np.full(8, 0.5, dtype=np.float32)
        assert _top_k_order(scores, 3).tolist() == [0, 1, 2]


class TestSearch:
    async def test_scores_descending_and_ranks_one_based(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        hits = await index.search(query_vector(), 5)

        assert [hit.fragment.fragment_id for hit in hits] == [
            "s2-0",
            "s2-1",
            "s2-2",
            "s1-0",
            "s1-1",
        ]
        assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
        assert [hit.origin for hit in hits] == ["dense"] * 5
        assert [hit.score for hit in hits] == pytest.approx(
            [0.99, 0.98, 0.97, 0.90, 0.80], abs=1e-5
        )
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(isinstance(hit, ScoredFragment) for hit in hits)

    async def test_allowed_keys_filters_before_top_k(self) -> None:
        """在被限定范围内必须仍能取满 k 条——先全局取 top-k 再过滤会返回 0 条。"""
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        # 全局 top-2 全部来自 s2，若实现是"先取 top-k 再过滤"，限定到 s1 会得到空结果。
        global_hits = await index.search(query_vector(), 2)
        assert {hit.fragment.source_key for hit in global_hits} == {Corpus.KEYS[1]}

        scoped = await index.search(query_vector(), 2, allowed_keys=[Corpus.KEYS[0]])

        assert len(scoped) == 2, "过滤后仍必须返回 min(k, 范围内候选数) 条"
        assert {hit.fragment.source_key for hit in scoped} == {Corpus.KEYS[0]}
        assert [hit.fragment.fragment_id for hit in scoped] == ["s1-0", "s1-1"]
        assert [hit.rank for hit in scoped] == [1, 2]
        assert [hit.score for hit in scoped] == pytest.approx([0.90, 0.80], abs=1e-5)

    async def test_allowed_keys_accepts_any_collection(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        hits = await index.search(query_vector(), 3, allowed_keys={Corpus.KEYS[2]})
        assert len(hits) == 3
        assert {hit.fragment.source_key for hit in hits} == {Corpus.KEYS[2]}

    async def test_allowed_keys_outside_index_returns_empty(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        assert await index.search(query_vector(), 3, allowed_keys=["unknown-key"]) == []

    @pytest.mark.parametrize("k", [0, -1, -100])
    async def test_non_positive_k_returns_empty(self, k: int) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        assert await index.search(query_vector(), k) == []

    async def test_k_larger_than_index_returns_everything(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        hits = await index.search(query_vector(), 100)

        assert len(hits) == len(corpus.fragments)
        assert [hit.rank for hit in hits] == list(range(1, len(corpus.fragments) + 1))
        # k ≥ 索引大小时走的是另一条排序分支，同样必须降序——否则"返回条数对了、
        # 顺序错了"这种缺陷不会被任何断言发现。
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert hits[0].fragment.fragment_id == "s2-0"
        assert hits[-1].score == pytest.approx(0.0, abs=1e-6)

    async def test_top_k_is_ordered_even_when_it_is_not_a_prefix(self) -> None:
        """得分顺序与入库顺序不同时必须按得分排序。

        这条测试专门盯住"候选数 > k"的分支：那里用 ``argpartition`` 选人，
        而 partition **只保证选出最大的 k 个、不保证它们的顺序**。若语料的得分顺序
        恰好与入库顺序一致（很容易在测试里无意中构造出来），漏掉那一步排序也不会
        被任何断言发现。
        """
        cosines = (0.50, 0.95, 0.60, 0.99, 0.95, 0.90)
        index = NumpyVectorIndex()
        await index.add(
            [
                (
                    make_fragment(f"shuffled-{rank}", Corpus.KEYS[0], chunk_index=rank),
                    vector_with_cosine(value, axis=rank + 1),
                )
                for rank, value in enumerate(cosines)
            ]
        )

        hits = await index.search(query_vector(), 4)

        # 0.95 出现两次（第 1、4 行）：并列时按入库顺序，行号小者在前。
        assert [hit.fragment.fragment_id for hit in hits] == [
            "shuffled-3",
            "shuffled-1",
            "shuffled-4",
            "shuffled-5",
        ]
        assert [hit.score for hit in hits] == pytest.approx([0.99, 0.95, 0.95, 0.90], abs=1e-6)
        assert [hit.rank for hit in hits] == [1, 2, 3, 4]

    async def test_query_dimension_mismatch_raises(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        with pytest.raises(ValueError):
            await index.search(normalize(basis(0, dim=4)), 1)

    async def test_query_need_not_be_normalized(self) -> None:
        """查询未归一化只缩放得分，不改变排序（实现刻意不因此报错）。"""
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        hits = await index.search(2.0 * query_vector(), 3)
        assert [hit.fragment.fragment_id for hit in hits] == ["s2-0", "s2-1", "s2-2"]
        assert [hit.score for hit in hits] == pytest.approx([1.98, 1.96, 1.94], abs=1e-5)


class TestRemove:
    async def test_remove_returns_count_and_hides_from_search(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        removed = await index.remove(Corpus.KEYS[1])

        assert removed == 3
        assert len(index) == len(corpus.fragments) - 3
        hits = await index.search(query_vector(), 100)
        assert all(hit.fragment.source_key != Corpus.KEYS[1] for hit in hits)
        assert "s2-0" not in {hit.fragment.fragment_id for hit in hits}
        assert index.dimension == DIM, "删除不应影响维度"

    async def test_remove_unknown_key_returns_zero(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        assert await index.remove("no-such-key") == 0
        assert len(index) == len(corpus.fragments)

    async def test_remove_all_then_re_add(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        for key in Corpus.KEYS:
            await index.remove(key)

        assert len(index) == 0
        assert index.fragments() == []
        assert await index.search(query_vector(), 5) == []

        # 清空后重新入库：行号映射必须被正确重建。
        await index.add(corpus.items())
        assert len(index) == len(corpus.fragments)
        assert (await index.search(query_vector(), 1))[0].fragment.fragment_id == "s2-0"


class TestMmrSearch:
    async def test_mmr_promotes_diversity(self) -> None:
        """MMR 的第 2 名应当是离群片段，从而显著降低结果集内的最大两两相似度。"""
        query, items, outlier_id = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)
        vector_of = {fragment.fragment_id: vector for fragment, vector in items}

        dense = await index.search(query, 2)
        diverse = await index.mmr_search(query, 2, fetch_k=6, lambda_=0.5)

        dense_ids = [hit.fragment.fragment_id for hit in dense]
        diverse_ids = [hit.fragment.fragment_id for hit in diverse]

        assert all(fragment_id.startswith("cluster-") for fragment_id in dense_ids)
        assert outlier_id not in dense_ids
        assert diverse_ids[0] == dense_ids[0], "首位仍应是最相关的片段"
        assert outlier_id in diverse_ids, "MMR 必须用多样性换掉一个近重复块"
        assert [hit.origin for hit in diverse] == ["mmr", "mmr"]
        assert [hit.rank for hit in diverse] == [1, 2]

        dense_overlap = max_pairwise_cosine([vector_of[i] for i in dense_ids])
        diverse_overlap = max_pairwise_cosine([vector_of[i] for i in diverse_ids])
        assert dense_overlap > 0.99
        assert diverse_overlap < 0.6
        assert diverse_overlap < dense_overlap - 0.3

    async def test_lambda_one_matches_plain_search(self) -> None:
        """λ=1 时 MMR 退化为纯相似度检索：结果与得分都应一致。"""
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        dense = await index.search(query, 3)
        mmr = await index.mmr_search(query, 3, fetch_k=len(items), lambda_=1.0)

        assert [hit.fragment.fragment_id for hit in mmr] == [
            hit.fragment.fragment_id for hit in dense
        ]
        assert [hit.score for hit in mmr] == pytest.approx([hit.score for hit in dense], abs=1e-6)

    async def test_mmr_scores_follow_the_spec_formula(self) -> None:
        """首位惩罚项为 0，故其 MMR 得分应恰好等于 λ·cos(q,d)。"""
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)
        vector_of = {fragment.fragment_id: vector for fragment, vector in items}

        hits = await index.mmr_search(query, 1, fetch_k=6, lambda_=0.5)
        expected = 0.5 * cosine(query, vector_of[hits[0].fragment.fragment_id])
        assert hits[0].score == pytest.approx(expected, abs=1e-6)

    async def test_mmr_ranks_are_contiguous(self) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        hits = await index.mmr_search(query, 4, fetch_k=6)
        assert [hit.rank for hit in hits] == [1, 2, 3, 4]
        assert len({hit.fragment.fragment_id for hit in hits}) == 4, "同一条不应重复入选"

    async def test_fetch_k_smaller_than_k_does_not_crash(self) -> None:
        query, items, outlier_id = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        hits = await index.mmr_search(query, 3, fetch_k=1)

        assert len(hits) == 3, "fetch_k 被提升为 k 后仍应返回 k 条"
        assert [hit.rank for hit in hits] == [1, 2, 3]
        # 候选池被提升为 k=3，即"相关性最高的 3 个"（全是簇内近重复），
        # 离群片段（相关性第 6）不应出现——这同时钉住了"候选池确实按相关性截断"。
        assert all(hit.fragment.fragment_id.startswith("cluster-") for hit in hits)

    async def test_fetch_k_limits_the_candidate_pool(self) -> None:
        """``fetch_k`` 是硬约束：池子之外的片段无论多"多样"都不该被选中。"""
        query, items, outlier_id = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        # 池 = 相关性最高的 2 个（两个簇内成员），离群片段排名第 6，进不了池子。
        hits = await index.mmr_search(query, 2, fetch_k=2)

        assert [hit.fragment.fragment_id for hit in hits] == ["cluster-0", "cluster-1"]
        assert outlier_id not in {hit.fragment.fragment_id for hit in hits}

    async def test_default_fetch_k_is_four_times_k(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        with_fetch = await index.mmr_search(query_vector(), 2, fetch_k=8)
        default_fetch = await index.mmr_search(query_vector(), 2)

        assert [hit.fragment.fragment_id for hit in default_fetch] == [
            hit.fragment.fragment_id for hit in with_fetch
        ]

    async def test_mmr_k_larger_than_candidates_returns_all(self) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        hits = await index.mmr_search(query, 50, fetch_k=50)
        assert len(hits) == len(items)

    async def test_mmr_respects_allowed_keys(self) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        hits = await index.mmr_search(query, 2, fetch_k=6, allowed_keys=[Corpus.KEYS[1]])

        assert len(hits) == 1, "范围内只有离群片段一条"
        assert hits[0].fragment.fragment_id == "outlier"

    async def test_mmr_with_empty_allowed_keys_returns_empty(self) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        assert await index.mmr_search(query, 2, fetch_k=6, allowed_keys=[]) == []

    @pytest.mark.parametrize("k", [0, -3])
    async def test_mmr_non_positive_k_returns_empty(self, k: int) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        assert await index.mmr_search(query, k, fetch_k=6) == []

    async def test_mmr_rejects_out_of_range_lambda(self) -> None:
        query, items, _ = mmr_cluster_items()
        index = NumpyVectorIndex()
        await index.add(items)

        with pytest.raises(ValueError):
            await index.mmr_search(query, 2, fetch_k=6, lambda_=1.5)
        with pytest.raises(ValueError):
            await index.mmr_search(query, 2, fetch_k=6, lambda_=-0.1)


class TestPersistence:
    async def test_persist_writes_expected_files(self, tmp_path: Path) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        assert (tmp_path / "vectors.npy").is_file()
        assert (tmp_path / "fragments.jsonl").is_file()

        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["dimension"] == DIM
        assert meta["n_fragments"] == len(corpus.fragments)

        lines = [
            line
            for line in (tmp_path / "fragments.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        assert len(lines) == len(corpus.fragments)
        assert Fragment.model_validate_json(lines[0]) == corpus.fragments[0]
        assert np.load(tmp_path / "vectors.npy").shape == (len(corpus.fragments), DIM)

    async def test_roundtrip_preserves_search_results(self, tmp_path: Path) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        restored = NumpyVectorIndex.from_path(tmp_path)

        assert len(restored) == len(index)
        assert restored.dimension == DIM
        assert [fragment.fragment_id for fragment in restored.fragments()] == [
            fragment.fragment_id for fragment in index.fragments()
        ]

        before = await index.search(query_vector(), 5)
        after = await restored.search(query_vector(), 5)
        assert [hit.fragment.fragment_id for hit in after] == [
            hit.fragment.fragment_id for hit in before
        ]
        assert [hit.rank for hit in after] == [hit.rank for hit in before]
        assert [hit.score for hit in after] == pytest.approx(
            [hit.score for hit in before], abs=1e-6
        )

    async def test_roundtrip_preserves_mmr_and_scoping(self, tmp_path: Path) -> None:
        query, items, outlier_id = mmr_cluster_items()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(items)
        await index.persist()

        restored = NumpyVectorIndex.from_path(tmp_path)

        hits = await restored.mmr_search(query, 2, fetch_k=6, lambda_=0.5)
        assert outlier_id in {hit.fragment.fragment_id for hit in hits}
        scoped = await restored.search(query, 2, allowed_keys=[Corpus.KEYS[0]])
        assert len(scoped) == 2

    async def test_roundtrip_of_empty_index_keeps_dimension(self, tmp_path: Path) -> None:
        index = NumpyVectorIndex(dimension=8, path=tmp_path)
        await index.persist()

        restored = NumpyVectorIndex.from_path(tmp_path)
        assert len(restored) == 0
        assert restored.dimension == 8, "空索引的维度也必须能往返"

    async def test_persist_without_path_is_noop(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())
        await index.persist()  # 不应抛异常，也不应落盘到任何位置
        assert len(index) == len(corpus.fragments)

    async def test_load_missing_path_returns_empty_index(self, tmp_path: Path) -> None:
        missing = tmp_path / "not-built-yet"

        index = NumpyVectorIndex.from_path(missing)

        assert len(index) == 0
        assert index.dimension == 0
        assert await index.search(query_vector(), 3) == []

    async def test_load_directory_without_vectors_returns_empty_index(self, tmp_path: Path) -> None:
        (tmp_path / "fragments.jsonl").write_text("", encoding="utf-8")
        index = NumpyVectorIndex(path=tmp_path).load()
        assert len(index) == 0

    async def test_load_into_configured_index_replaces_state(self, tmp_path: Path) -> None:
        corpus = Corpus()
        source = NumpyVectorIndex(path=tmp_path)
        await source.add(corpus.items())
        await source.persist()

        index = NumpyVectorIndex(dimension=DIM, path=tmp_path)
        await index.add([(make_fragment("stale", Corpus.KEYS[0]), normalize(basis(5)))])
        index.load()

        assert len(index) == len(corpus.fragments)
        assert "stale" not in {fragment.fragment_id for fragment in index.fragments()}

    async def test_load_rejects_dimension_mismatch(self, tmp_path: Path) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        with pytest.raises(ValueError):
            NumpyVectorIndex(dimension=4, path=tmp_path).load()

    async def test_load_rejects_meta_dimension_conflict(self, tmp_path: Path) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        (tmp_path / "meta.json").write_text(json.dumps({"dimension": 4}), encoding="utf-8")
        with pytest.raises(ValueError):
            NumpyVectorIndex.from_path(tmp_path)

    async def test_load_rejects_row_count_mismatch(self, tmp_path: Path) -> None:
        """矩阵行数与片段条数不一致意味着行号会错位，必须报错而不是继续。"""
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        lines = (tmp_path / "fragments.jsonl").read_text(encoding="utf-8").splitlines()
        (tmp_path / "fragments.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with pytest.raises(ValueError):
            NumpyVectorIndex.from_path(tmp_path)

    async def test_load_reports_corrupt_fragment_line(self, tmp_path: Path) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex(path=tmp_path)
        await index.add(corpus.items())
        await index.persist()

        with (tmp_path / "fragments.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{not json}\n")

        with pytest.raises(ValueError, match="fragments.jsonl"):
            NumpyVectorIndex.from_path(tmp_path)


class TestClearAndEmptyIndex:
    async def test_clear_empties_index_but_keeps_dimension(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        index.clear()

        assert len(index) == 0
        assert index.fragments() == []
        assert index.dimension == DIM
        assert await index.search(query_vector(), 3) == []
        assert await index.remove(Corpus.KEYS[0]) == 0

    async def test_all_methods_safe_on_empty_index(self) -> None:
        index = NumpyVectorIndex()

        assert len(index) == 0
        assert index.dimension == 0
        assert index.fragments() == []
        assert await index.search(query_vector(), 5) == []
        assert await index.search(query_vector(), 5, allowed_keys=[Corpus.KEYS[0]]) == []
        assert await index.mmr_search(query_vector(), 5, fetch_k=20) == []
        assert await index.remove(Corpus.KEYS[0]) == 0
        index.clear()
        await index.persist()

    async def test_fragments_returns_a_copy(self) -> None:
        corpus = Corpus()
        index = NumpyVectorIndex()
        await index.add(corpus.items())

        snapshot = index.fragments()
        snapshot.clear()

        assert len(index) == len(corpus.fragments)
        assert [fragment.fragment_id for fragment in index.fragments()] == [
            fragment.fragment_id for fragment in corpus.fragments
        ]

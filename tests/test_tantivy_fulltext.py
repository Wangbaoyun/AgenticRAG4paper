"""``TantivyFullTextIndex`` 的行为测试。

分三组钉住不变式：

1. **切分一致性**（本适配器最危险的失效模式）：入库与查询两侧必须都走
   ``to_index_terms``。这条一旦破掉，检索不会报错，只会静默零召回——
   所以既钉行为（随机子串必命中、未登录术语必命中），也钉字节
   （``text_zh`` 的 STORED 值必须等于 ``to_index_terms(text)``）。
2. **查询语义**：正文取 max、标题加权相加、``allowed_keys`` 在索引内过滤、
   英文词形归一。
3. **契约**：缓冲写入、删除计数、``rank`` / 条数、``Fragment`` 完整还原、退化输入。

测试数据（作者名、标题、DOI 一律虚构）与项目其他测试保持一致：不使用任何真实
论文数据，避免审计脚本产生无意义的相似命中。每个测试用独立的 ``tmp_path`` 索引目录，
测试之间不共享状态。
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
import tantivy

from scitrace.adapters.indexes.tantivy_fulltext import _TITLE_BOOST, TantivyFullTextIndex
from scitrace.domain import Fragment
from scitrace.domain.fragment import make_fragment_id
from scitrace.ports import FullTextIndex, ScoredFragment
from scitrace.util import sha256_hex, to_index_terms, tokenize_mixed

#: 三篇**虚构**文献的键（16 位十六进制，符合 SourceKey 的形式约定）。
KEY_A = "0a1b2c3d4e5f6071"
KEY_B = "1b2c3d4e5f607182"
KEY_C = "2c3d4e5f60718293"

#: 中英混排语料。用真实感的中英混排而非纯中文，才能同时压到两条切分路径。
_MIXED_TEXT = (
    "本文提出一种跨模态对齐方法，在 RAG 系统与稀疏专家路由上取得 87.5% 的准确率，"
    "we call it Sparse Expert Routing and evaluate it on three multilingual benchmarks."
)


# --------------------------------------------------------------------- 夹具与工具


@pytest.fixture
def index(tmp_path: Path) -> TantivyFullTextIndex:
    """指向临时目录的空索引。"""
    return TantivyFullTextIndex(tmp_path / "fulltext")


def _fragment(
    source_key: str,
    text: str,
    *,
    chunk_index: int = 0,
    section_path: list[str] | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    media: list[dict[str, object]] | None = None,
) -> Fragment:
    """构造一个字段齐全的片段，``fragment_id`` 走领域层的确定性派生。"""
    section = ["2 Method"] if section_path is None else section_path
    return Fragment(
        fragment_id=make_fragment_id(
            source_key=source_key,
            # 真实摄入传的是文件内容哈希，这里用文本哈希代替，保持同一形态
            document_hash=sha256_hex(text),
            section_path=section,
            chunk_index=chunk_index,
        ),
        source_key=source_key,
        text=text,
        chunk_index=chunk_index,
        section_path=section,
        page_start=page_start,
        page_end=page_end,
        char_count=len(text),
        media=[] if media is None else media,
    )


async def _ingest(
    index: TantivyFullTextIndex,
    fragments: Sequence[Fragment],
    *,
    titles: dict[str, str] | None = None,
) -> None:
    """写入并提交（大多数测试只关心提交后的世界）。"""
    await index.add(fragments, titles=titles)
    await index.commit()


def _ids(hits: Sequence[ScoredFragment]) -> list[str]:
    """把检索结果压成 fragment_id 列表，便于断言顺序与集合。"""
    return [hit.fragment.fragment_id for hit in hits]


def _field_scores(
    index: TantivyFullTextIndex, field_name: str, query: str
) -> dict[str, float]:
    """用**同一个** searcher 分别跑单字段查询，取回每个片段的单字段 BM25 分。

    这是白盒断言：适配器规定"正文取 max、标题加权相加"，只有在同一份语料统计量下
    把各字段的分数分别取出来，才能验证合并规则真的按文档说的方式发生。
    """
    searcher = index._index.searcher()  # 白盒：合并规则本身就是要被验证的对象
    terms = to_index_terms(query)
    field_query = index._index.parse_query(terms, default_field_names=[field_name])
    hits = searcher.search(field_query, max(searcher.num_docs, 1)).hits
    return {
        str(searcher.doc(address).get_first("fragment_id")): float(score)
        for score, address in hits
    }


def _token_safe_windows(text: str, *, count: int, seed: int) -> list[str]:
    """从 ``text`` 中取若干**不切断词**的子串当查询。

    切点必须落在 token 边界上：在英文单词中间切开会产生一个索引里根本不存在的
    token（"retriev"），在数字组中间拼接会造出原文没有的 token（"87" + "5" → "875"）。
    那些失败是切分工具应有的行为，不该算在"入库/查询一致性"头上。
    做法是按最大字母数字/CJK 连续段取跨度，再按原文字符区间切片。
    """
    spans = [match.span() for match in re.finditer(r"[0-9A-Za-z\u4e00-\u9fff]+", text)]
    rng = random.Random(seed)
    windows: list[str] = []
    for _ in range(count):
        start = rng.randrange(len(spans))
        end = rng.randrange(start + 1, len(spans) + 1)
        windows.append(text[spans[start][0] : spans[end - 1][1]])
    return windows


# ------------------------------------------------------- 1. 切分一致性（最重要）


async def test_query_and_index_share_the_same_tokenization(index: TantivyFullTextIndex) -> None:
    """回归测试：随机子串查询必须命中原文。

    若入库侧写了原文而查询侧做了 bigram（或反之），这里会大面积零召回——
    这是唯一能在**没有**白盒检查时抓住该缺陷的测试。
    """
    fragment = _fragment(KEY_A, _MIXED_TEXT)
    await _ingest(index, [fragment])

    windows = _token_safe_windows(_MIXED_TEXT, count=16, seed=20240607)
    assert len(set(windows)) >= 10, "样本太单一，覆盖不到中英混排的多种切点"

    for window in windows:
        # 前提：窗口的 token 全都在原文的 token 里（窗口没有把词切开）
        assert set(tokenize_mixed(window)) <= set(tokenize_mixed(_MIXED_TEXT)), window
        hits = await index.search(window, 5)
        assert _ids(hits) == [fragment.fragment_id], (
            f"查询 {window!r} 未命中原文：入库与查询两侧的切分大概率不一致"
        )


async def test_index_side_stores_bigram_form(index: TantivyFullTextIndex) -> None:
    """字节级钉子：``text_zh`` 存的必须**恰好**是 ``to_index_terms(text)``。

    只测行为有可能被"两侧都错但错得一致"蒙混过去（例如两侧都忘了归一化），
    所以这里直接核对索引里的值。
    """
    text = "本文提出跨模态对齐方法，并在 RAG 系统上验证 retrieval augmented generation。"
    fragment = _fragment(KEY_A, text)
    await _ingest(index, [fragment])

    searcher = index._index.searcher()  # 白盒：刻意核对索引里到底存了什么
    address = searcher.search(index._term_query("source_key", KEY_A), 5).hits[0][1]
    document = searcher.doc(address)
    assert document.get_first("text_zh") == to_index_terms(fragment.text)
    # 与领域模型归一化后的文本比较（Fragment 的校验器会做 NFKC，全角逗号会变半角）
    assert document.get_first("text_en") == fragment.text
    assert document.get_first("text_en") != to_index_terms(fragment.text)
    # 反向对照：中文查询本身不是索引词，必须先转 bigram——否则静默零召回
    assert to_index_terms("跨模态") == "跨模 模态"


async def test_chinese_bigram_recall(index: TantivyFullTextIndex) -> None:
    """中文召回：短查询与整短语都要命中（bigram 的容错性）。"""
    fragment = _fragment(KEY_A, "本文提出一种跨模态对齐方法，用于多语言检索。")
    await _ingest(index, [fragment])

    for query in ("跨模态", "跨模态对齐", "对齐方法", "多语言检索"):
        hits = await index.search(query, 5)
        assert _ids(hits) == [fragment.fragment_id], f"{query} 应命中"


async def test_unseen_technical_term_recall(index: TantivyFullTextIndex) -> None:
    """未登录术语：语料里没有词典条目，bigram 也必须召回。"""
    fragment = _fragment(KEY_A, "我们引入稀疏专家路由机制来降低推理成本。")
    await _ingest(index, [fragment])

    for query in ("稀疏专家", "专家路由", "路由机制", "稀疏专家路由"):
        assert _ids(await index.search(query, 5)) == [fragment.fragment_id], query


async def test_english_recall_with_stemming(index: TantivyFullTextIndex) -> None:
    """英文召回与词形归一（``en_stem``）：不同词形必须命中同一片段。"""
    fragment = _fragment(KEY_A, "Retrieval Augmented Generation improves question answering.")
    await _ingest(index, [fragment])

    for query in ("retrieval", "generations", "generation", "augmenting", "augmentation"):
        assert _ids(await index.search(query, 5)) == [fragment.fragment_id], query


async def test_mixed_script_does_not_interfere(index: TantivyFullTextIndex) -> None:
    """中英混排：中文查询与英文查询互不干扰，各自都能命中同一片段。"""
    fragment = _fragment(KEY_A, "RAG 系统在多语言检索上的表现")
    await _ingest(index, [fragment])

    for query in ("RAG", "rag", "系统", "多语言检索"):
        assert _ids(await index.search(query, 5)) == [fragment.fragment_id], query


async def test_single_character_query_is_a_documented_limitation(
    index: TantivyFullTextIndex,
) -> None:
    """bigram 的已知代价：单字查询命中不了多字词中间的字。

    这是"免词典、宽召回"这一取舍的必然结果（"跨模态"只产出 ``跨模``/``模态``；
    孤立的 ``态`` 不是索引词）。项目用两段式检索补偿精度（SPEC §3.6），
    若将来需要单字召回，应加 ngram 前缀索引而不是退回词典分词。
    """
    fragment = _fragment(KEY_A, "本文提出一种跨模态对齐方法。")
    await _ingest(index, [fragment])

    assert _ids(await index.search("跨模", 5)) == [fragment.fragment_id]
    # 词首字命中（"跨" 作为单字段落出现时才会成为索引词），中间字不命中
    assert await index.search("态", 5) == []


# ------------------------------------------------------------- 2. 查询语义


async def test_body_score_is_max_of_zh_and_en(index: TantivyFullTextIndex) -> None:
    """正文得分取 ``max(score_zh, score_en)``，不是两者相加。"""
    fragment = _fragment(KEY_A, "检索 retrieval")
    await _ingest(index, [fragment])

    zh_score = _field_scores(index, "text_zh", "retrieval")[fragment.fragment_id]
    en_score = _field_scores(index, "text_en", "retrieval")[fragment.fragment_id]
    assert zh_score > 0 and en_score > 0, "本用例要两个字段都命中才有鉴别力"

    hits = await index.search("retrieval", 5)
    assert hits[0].score == pytest.approx(max(zh_score, en_score), rel=1e-9)
    # 若实现把多字段得分相加，下面这条会失败
    assert hits[0].score < zh_score + en_score


async def test_title_score_is_additive(index: TantivyFullTextIndex) -> None:
    """标题与正文的合并是"加权相加"：同正文同标题的片段得分可分解。

    ``A`` 只有正文命中、``B`` 只有标题命中、``C`` 两者都命中，
    则 ``score(C) == score(A) + score(B)``。这条同时证明了"只命中标题"的片段
    也能被召回——纯 max 的实现会让 C 的得分等于 max(A, B) 而不是两者之和。
    """
    body = "We propose a crossmodal alignment method for scientific retrieval."
    other_body = "本文讨论知识图谱的构建与维护流程。"
    title = "Crossmodal Alignment for Evidence Tracing"
    frag_body = _fragment(KEY_A, body)
    frag_title = _fragment(KEY_B, other_body)
    frag_both = _fragment(KEY_C, body)
    await _ingest(index, [frag_body, frag_title, frag_both], titles={KEY_B: title, KEY_C: title})

    hits = await index.search("crossmodal", 10)
    scores = {hit.fragment.source_key: hit.score for hit in hits}
    assert set(scores) == {KEY_A, KEY_B, KEY_C}
    assert scores[KEY_A] > 0 and scores[KEY_B] > 0
    assert scores[KEY_C] == pytest.approx(scores[KEY_A] + scores[KEY_B], rel=1e-9)

    # 白盒复核：正文取 max + 标题按 _TITLE_BOOST 加权
    expected = max(
        _field_scores(index, "text_zh", "crossmodal").get(frag_both.fragment_id, 0.0),
        _field_scores(index, "text_en", "crossmodal").get(frag_both.fragment_id, 0.0),
    ) + _TITLE_BOOST * _field_scores(index, "title", "crossmodal")[frag_both.fragment_id]
    assert scores[KEY_C] == pytest.approx(expected, rel=1e-9)


async def test_title_words_find_the_paper(index: TantivyFullTextIndex) -> None:
    """标题检索：中英文标题都要能按词命中（标题写的是 bigram 串 + en_stem）。"""
    frags_a = [_fragment(KEY_A, f"本文讨论对齐方法的第 {i} 个变体。", chunk_index=i) for i in range(2)]
    frags_b = [_fragment(KEY_B, f"实验部分给出第 {i} 组基准结果。", chunk_index=i) for i in range(2)]
    await _ingest(
        index,
        frags_a + frags_b,
        titles={
            KEY_A: "Sparse Expert Routing for Evidence Tracing",
            KEY_B: "面向科研文献的证据可溯源问答方法",
        },
    )

    english = await index.search("evidence tracing", 10)
    assert {hit.fragment.source_key for hit in english} == {KEY_A}
    chinese = await index.search("证据可溯源", 10)
    assert {hit.fragment.source_key for hit in chinese} == {KEY_B}
    # 标题只命中时也要召回（正文里没有这个词）
    assert len(await index.search("tracing", 10)) == len(frags_a)


async def test_fragment_without_title_is_not_matched_by_title_index(
    index: TantivyFullTextIndex,
) -> None:
    """未提供标题的文献不会被标题查询误召回。"""
    frags_a = [_fragment(KEY_A, "正文不包含那个词。")]
    frags_c = [_fragment(KEY_C, "另一篇的正文。")]
    await _ingest(index, frags_a + frags_c, titles={KEY_C: "Unrelated Bibliographic Record"})

    assert await index.search("bibliographic", 10) != []
    assert _ids(await index.search("bibliographic", 10)) == [frags_c[0].fragment_id]


async def test_allowed_keys_filters_inside_the_query(index: TantivyFullTextIndex) -> None:
    """``allowed_keys`` 必须在索引内过滤，不能先取 top-k 再筛。"""
    query = "跨模态对齐"
    # 短文本 → BM25 更高，全局前列被 KEY_B 占满
    fillers = [_fragment(KEY_B, "跨模态对齐。", chunk_index=i) for i in range(6)]
    # 长文本且只出现一次 → 全局排序靠后
    wanted = [
        _fragment(KEY_A, "跨模态对齐。" + "本文在多个数据集上做了一系列消融实验并给出详细分析。" * 5,
                  chunk_index=i)
        for i in range(3)
    ]
    await _ingest(index, fillers + wanted)

    top2 = await index.search(query, 2)
    assert {hit.fragment.source_key for hit in top2} == {KEY_B}, "用例前提：全局前 2 名都在 KEY_B"

    filtered = await index.search(query, 2, allowed_keys={KEY_A})
    assert len(filtered) == 2, "先取 top-k 再过滤的实现会在这里返回 0 条"
    assert {hit.fragment.source_key for hit in filtered} == {KEY_A}
    assert [hit.rank for hit in filtered] == [1, 2]

    assert len(await index.search(query, 99, allowed_keys=[KEY_A])) == 3
    assert len(await index.search(query, 99, allowed_keys=(KEY_A, KEY_B))) == 9


async def test_allowed_keys_empty_or_unmatched_returns_nothing(
    index: TantivyFullTextIndex,
) -> None:
    """空集合返回空列表；集合里没有命中的文献也返回空列表（且不抛异常）。"""
    await _ingest(index, [_fragment(KEY_A, "跨模态对齐方法。")])

    assert await index.search("跨模态", 5, allowed_keys=set()) == []
    assert await index.search("跨模态", 5, allowed_keys=[KEY_B]) == []
    assert len(await index.search("跨模态", 5, allowed_keys=[KEY_A])) == 1


# ------------------------------------------------------------------ 3. 契约


async def test_add_is_buffered_until_commit(index: TantivyFullTextIndex) -> None:
    """``add`` 之后必须 ``commit`` 才对 ``search`` / ``len`` 可见。"""
    await index.add([_fragment(KEY_A, "跨模态对齐方法。")])
    assert len(index) == 0
    assert await index.search("跨模态", 5) == []

    await index.commit()
    assert len(index) == 1
    assert len(await index.search("跨模态", 5)) == 1


async def test_repeated_add_of_same_fragment_id_overwrites(
    index: TantivyFullTextIndex,
) -> None:
    """同一 ``fragment_id`` 重复写入是覆盖，不留下陈旧副本。"""
    first = _fragment(KEY_A, "跨模态对齐的初版描述。")
    replacement_text = "稀疏专家路由的改进描述。"
    second = Fragment(
        fragment_id=first.fragment_id,
        source_key=first.source_key,
        text=replacement_text,
        chunk_index=first.chunk_index,
        section_path=first.section_path,
        char_count=len(replacement_text),
    )
    await _ingest(index, [first])
    await _ingest(index, [second])

    assert len(index) == 1, "重复写入不应产生两份文档"
    assert await index.search("跨模态", 5) == [], "旧内容必须随覆盖一起消失"
    hits = await index.search("稀疏专家路由", 5)
    assert [hit.fragment.text for hit in hits] == [replacement_text]


async def test_remove_deletes_all_fragments_of_a_source(index: TantivyFullTextIndex) -> None:
    """``remove`` 删除该文献全部片段并返回条数；提交后不再命中。"""
    frags_a = [_fragment(KEY_A, f"跨模态对齐方法的第一部分 {i}。", chunk_index=i) for i in range(3)]
    frags_b = [_fragment(KEY_B, f"跨模态检索在基准 {i} 上的表现。", chunk_index=i) for i in range(2)]
    await _ingest(index, frags_a + frags_b)
    assert len(index) == 5

    assert await index.remove(KEY_A) == 3
    # 删除是缓冲的：commit 之前重复删除不重复计数，len 也还看不到变化
    assert await index.remove(KEY_A) == 0
    assert len(index) == 5

    await index.commit()
    assert len(index) == 2
    hits = await index.search("跨模态", 10)
    assert {hit.fragment.source_key for hit in hits} == {KEY_B}

    assert await index.remove("ffffffffffffffff") == 0


async def test_scores_descend_ranks_are_contiguous_and_count_is_min(
    index: TantivyFullTextIndex,
) -> None:
    """得分降序、``rank`` 从 1 连续递增、条数 = ``min(k, 命中数)``。"""
    fragments = [
        _fragment(KEY_A, f"跨模态对齐方法第 {i} 节的实验细节与消融分析结论。", chunk_index=i)
        for i in range(5)
    ]
    await _ingest(index, fragments)

    top3 = await index.search("跨模态", 3)
    assert len(top3) == 3
    assert [hit.rank for hit in top3] == [1, 2, 3]
    scores = [hit.score for hit in top3]
    assert scores == sorted(scores, reverse=True)
    assert all(hit.origin == "bm25" for hit in top3)
    assert all(hit.fragment.source_key == KEY_A for hit in top3)
    assert scores[0] > 0

    assert len(await index.search("跨模态", 5)) == 5
    assert len(await index.search("跨模态", 99)) == 5


async def test_restored_fragment_is_complete(index: TantivyFullTextIndex) -> None:
    """检索结果里的 ``Fragment`` 必须与入库时逐字段一致（含 media 等 schema 外字段）。"""
    original = _fragment(
        KEY_A,
        "本文提出一种跨模态对齐方法，并在三个基准上验证其有效性。",
        chunk_index=7,
        section_path=["2 Method", "2.1 Cross-Modal Alignment"],
        page_start=3,
        page_end=4,
        media=[{"type": "figure", "caption": "虚构的图 1：对齐流程"}],
    )
    await _ingest(index, [original])

    hits = await index.search("跨模态对齐", 5)
    assert len(hits) == 1
    restored = hits[0].fragment
    # 整体相等 + 逐项点名：整体相等可能被 pydantic 的默认值掩盖字段级问题
    assert restored == original
    assert restored.fragment_id == original.fragment_id
    assert restored.source_key == KEY_A
    assert restored.text == original.text
    assert restored.chunk_index == 7
    assert restored.section_path == ["2 Method", "2.1 Cross-Modal Alignment"]
    assert (restored.page_start, restored.page_end) == (3, 4)
    assert restored.char_count == original.char_count == len(original.text)
    assert restored.media == [{"type": "figure", "caption": "虚构的图 1：对齐流程"}]
    assert hits[0].rank == 1


async def test_fragment_without_pages_round_trips_as_none(index: TantivyFullTextIndex) -> None:
    """没有页码的片段还原后仍是 ``None``，不会被伪造成 0。"""
    await _ingest(index, [_fragment(KEY_A, "跨模态对齐方法没有页码信息。")])

    hits = await index.search("跨模态", 5)
    assert hits[0].fragment.page_start is None
    assert hits[0].fragment.page_end is None


async def test_documents_without_json_payload_are_still_restorable(
    index: TantivyFullTextIndex,
) -> None:
    """降级路径：没有 ``fragment_json`` 的文档（旧版/手写内容）仍能还原成可读片段。"""
    text = "旧版写入的跨模态对齐片段"
    writer = index._index.writer()  # 白盒：刻意绕过适配器的写入路径
    document = tantivy.Document()
    document.add_text("fragment_id", "deadbeefdeadbeef")
    document.add_text("source_key", KEY_A)
    document.add_text("text_zh", to_index_terms(text))
    document.add_text("text_en", text)
    document.add_text("section_path", "1 Introduction")
    document.add_integer("page_start", 2)
    writer.add_document(document)
    writer.commit()
    index._index.reload()

    hits = await index.search("跨模态", 5)
    assert len(hits) == 1
    restored = hits[0].fragment
    assert restored.fragment_id == "deadbeefdeadbeef"
    assert restored.text == text
    assert restored.section_path == ["1 Introduction"]
    assert restored.page_start == 2
    assert restored.page_end is None
    assert restored.chunk_index == 0
    assert restored.char_count == len(text)


async def test_degenerate_inputs_return_empty_lists(index: TantivyFullTextIndex) -> None:
    """空索引、``k <= 0``、空查询串、纯标点查询都返回空列表且不抛异常。"""
    assert await index.search("跨模态", 5) == []
    assert len(index) == 0

    await _ingest(index, [_fragment(KEY_A, "跨模态对齐方法。")])
    assert await index.search("跨模态", 0) == []
    assert await index.search("跨模态", -3) == []
    assert await index.search("", 5) == []
    assert await index.search("   ", 5) == []
    assert await index.search("！？，。…", 5) == []
    assert len(await index.search("跨模态", 1)) == 1


async def test_clear_empties_the_index(index: TantivyFullTextIndex) -> None:
    """``clear()`` 之后 ``len() == 0`` 且检索不到任何东西。"""
    await _ingest(index, [_fragment(KEY_A, f"跨模态对齐方法 {i}。", chunk_index=i) for i in range(3)])
    assert len(index) == 3

    index.clear()
    assert len(index) == 0
    assert await index.search("跨模态", 5) == []

    # 清空后仍可继续写入
    await _ingest(index, [_fragment(KEY_B, "稀疏专家路由方法。")])
    assert len(index) == 1
    assert len(await index.search("专家路由", 5)) == 1


async def test_index_directory_is_created_automatically(tmp_path: Path) -> None:
    """索引目录（含多级父目录）不存在时自动创建。"""
    nested = tmp_path / "indexes" / "demo" / "fulltext"
    assert not nested.exists()

    index = TantivyFullTextIndex(nested)
    assert nested.is_dir()
    await _ingest(index, [_fragment(KEY_A, "跨模态对齐方法。")])
    assert len(await index.search("跨模态", 5)) == 1


async def test_satisfies_fulltext_index_protocol(tmp_path: Path) -> None:
    """结构上满足 ``FullTextIndex`` 协议（方法名与签名漂移会在这里暴露）。"""
    assert isinstance(TantivyFullTextIndex(tmp_path / "idx"), FullTextIndex)

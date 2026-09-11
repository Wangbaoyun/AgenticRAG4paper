"""测试结构感知分块与句切分。

分块是摄入链路里最能影响最终质量的一步：块太大则检索精度下降，
块太小则证据失去上下文，不感知结构则引用只能给到页码而给不出章节。
本文件把这些性质逐条钉住。
"""

from __future__ import annotations

import pytest
from scitrace.config import ChunkingSettings
from scitrace.domain import ParsedDocument, ParsedPage
from scitrace.pipeline.chunking import (
    chunk_document,
    detect_heading,
    split_paragraphs,
    split_sentences,
    strip_references_section,
)


class TestSplitSentences:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("第一句。第二句！", ["第一句。", "第二句！"]),
            ("问题？答案。", ["问题？", "答案。"]),
            ("分号；也算。", ["分号；", "也算。"]),
            ("省略号……结束。", ["省略号……", "结束。"]),
        ],
    )
    def test_chinese_terminators(self, text: str, expected: list[str]) -> None:
        assert split_sentences(text) == expected

    def test_english_terminators(self) -> None:
        assert split_sentences("First one. Second one! Third?") == [
            "First one.",
            "Second one!",
            "Third?",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "See Fig. 3 for details. It works.",
            "Smith et al. reported this. Really.",
            "Use e.g. this one. It works.",
            "The value is 3.14 exactly. Done.",
            "J. Smith wrote it. Yes.",
        ],
    )
    def test_abbreviations_and_decimals_do_not_split(self, text: str) -> None:
        """缩写、小数、人名首字母后的句点不是句末——这是学术文本里的高频误切来源。"""
        assert len(split_sentences(text)) == 2

    def test_lowercase_continuation_is_not_a_boundary(self) -> None:
        """句点后接小写字母说明它并非句末（如编号后的续写）。"""
        assert split_sentences("See item 3. for details here.") == [
            "See item 3. for details here."
        ]

    def test_closing_quote_belongs_to_sentence(self) -> None:
        assert split_sentences('他说：“可以。”然后走了。') == ['他说：“可以。”', "然后走了。"]

    def test_mixed_language(self) -> None:
        assert split_sentences("本文提出 RAG 方法。It works well.") == [
            "本文提出 RAG 方法。",
            "It works well.",
        ]

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_empty_input(self, text: str) -> None:
        assert split_sentences(text) == []

    def test_text_without_terminator_is_one_sentence(self) -> None:
        assert split_sentences("没有句末标点的一段话") == ["没有句末标点的一段话"]


class TestDetectHeading:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("## Method", (2, "Method")),
            ("# Title", (1, "Title")),
            ("### 2.1 Details ###", (3, "2.1 Details")),
            ("2.1 Retrieval", (2, "2.1 Retrieval")),
            ("2 Method", (1, "2 Method")),
            ("IV. Results", (1, "IV. Results")),
            ("A. Proofs", (1, "A. Proofs")),
            ("第三章 方法", (1, "第三章 方法")),
            ("二、实验结果", (1, "二、实验结果")),
        ],
    )
    def test_recognized_forms(self, line: str, expected: tuple[int, str]) -> None:
        assert detect_heading(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "This is a normal sentence.",
            "3 samples were used in the experiment.",
            "I think this matters a lot",
            "2024 was a good year",
            "我们使用了 3 个样本进行实验。",
            "",
            "   ",
        ],
    )
    def test_body_text_is_not_a_heading(self, line: str) -> None:
        """宁缺毋滥：误判标题会把 section_path 变成噪声，比没有结构信息更糟。"""
        assert detect_heading(line) is None

    def test_all_caps_needs_blank_next_line(self) -> None:
        assert detect_heading("METHOD", next_line_is_blank=True) == (1, "METHOD")
        assert detect_heading("METHOD", next_line_is_blank=False) is None

    def test_overlong_line_is_not_a_heading(self) -> None:
        assert detect_heading("1 " + "x" * 200) is None


class TestSplitParagraphs:
    def test_blank_line_separates(self) -> None:
        assert split_paragraphs("one\n\ntwo\n\nthree") == ["one", "two", "three"]

    def test_soft_wrap_joined_with_space_for_latin(self) -> None:
        assert split_paragraphs("This is a\nwrapped line") == ["This is a wrapped line"]

    def test_soft_wrap_joined_without_space_for_cjk(self) -> None:
        """中文断行不能补空格：补了会切出跨词的噪声 bigram，直接损害检索召回。"""
        assert split_paragraphs("这是一个\n被换行拆开的中文段落") == [
            "这是一个被换行拆开的中文段落"
        ]

    def test_mixed_wrap(self) -> None:
        assert split_paragraphs("使用 RAG\n方法") == ["使用 RAG 方法"]

    def test_empty(self) -> None:
        assert split_paragraphs("") == []
        assert split_paragraphs("\n\n\n") == []


class TestStripReferences:
    def test_removes_reference_section(self) -> None:
        text = "Body text here.\n\nReferences\n\n[1] Someone. A paper. 2020.\n[2] Other. B paper.\n"
        assert "Someone" not in strip_references_section(text)
        assert "Body text here." in strip_references_section(text)

    @pytest.mark.parametrize("heading", ["References", "REFERENCES", "Bibliography", "参考文献"])
    def test_recognized_headings(self, heading: str) -> None:
        text = f"Body.\n\n{heading}\n\n[1] A very long reference entry that goes on and on.\n[2] Another.\n"
        assert "very long reference" not in strip_references_section(text)

    def test_inline_mention_is_kept(self) -> None:
        """正文中提到 "References" 的句子不能被当成区块起点。"""
        text = "See the References for details.\n\nMore body text."
        assert strip_references_section(text) == text

    def test_no_reference_section(self) -> None:
        text = "Just body text.\n\nSecond paragraph."
        assert strip_references_section(text) == text

    def test_appendix_after_references_is_kept(self) -> None:
        """参考文献之后还有附录时，附录**必须保留**。

        回归自真实数据：PaperQA2 原文正文只有 9 页，第 9 页末尾是 References，
        第 12–25 页是 "8 Methods / 8.1 ... / 8.2 LitQA" 等实质性附录。
        早先的实现"从 References 一路切到文末"，静默丢掉了全文 60% 的内容——
        而此前所有测试用的合成文档都把参考文献放在最后，完全测不到。
        """
        text = (
            "Body paragraph with real content.\n\n"
            "References\n\n"
            "[1] Someone. A paper title. 2020.\n"
            "[2] Other. Another paper title. 2021.\n\n"
            "8 Methods\n\n"
            "This appendix describes the implementation in detail.\n"
        )
        kept = strip_references_section(text)
        assert "This appendix describes" in kept, "附录被误删"
        assert "Another paper title" not in kept, "参考文献未被剔除"

    def test_second_reference_block_is_also_removed(self) -> None:
        """附录之后可能还有第二段参考文献（正文引用 + 附录引用分开列）。"""
        text = (
            "Body.\n\nReferences\n\n[1] First list entry with a year 2020.\n\n"
            "8 Methods\n\nAppendix content that matters.\n\n"
            "References\n\n[2] Second list entry with a year 2021.\n"
        )
        kept = strip_references_section(text)
        assert "Appendix content that matters." in kept
        assert "First list entry" not in kept
        assert "Second list entry" not in kept

    def test_reference_list_at_end_still_removed(self) -> None:
        """找不到后续标题时保持原行为：切到文末。"""
        text = "Body.\n\nReferences\n\n[1] Entry one.\n[2] Entry two.\n[3] Entry three.\n"
        kept = strip_references_section(text)
        assert "Entry one" not in kept
        assert "Body." in kept


def make_document(*pages: str, parser: str = "plaintext") -> ParsedDocument:
    return ParsedDocument(
        pages=[
            ParsedPage(page_number=index, text=text) for index, text in enumerate(pages, start=1)
        ],
        hints={},
        parser=parser,
    )


class TestChunkDocument:
    def test_empty_document(self) -> None:
        assert chunk_document(make_document(), source_key="s1") == []

    def test_basic_chunking(self) -> None:
        doc = make_document("1 Introduction\n\nAlpha paragraph one.\n\nAlpha paragraph two.")
        chunks = chunk_document(doc, source_key="s1")
        assert len(chunks) == 1
        assert chunks[0].section_path == ["1 Introduction"]
        assert "Alpha paragraph one." in chunks[0].text
        assert "Alpha paragraph two." in chunks[0].text

    def test_section_path_hierarchy(self) -> None:
        doc = make_document(
            "1 Method\n\nIntro text for method.\n\n1.1 Retrieval\n\nRetrieval details here."
        )
        chunks = chunk_document(doc, source_key="s1")
        assert chunks[0].section_path == ["1 Method"]
        assert chunks[-1].section_path == ["1 Method", "1.1 Retrieval"]

    def test_page_range_tracked(self) -> None:
        doc = make_document("Page one text that is long enough.", "Page two text that is longer.")
        chunks = chunk_document(doc, source_key="s1")
        assert chunks[0].page_start == 1
        assert chunks[-1].page_end == 2

    def test_target_chars_triggers_new_chunk(self) -> None:
        settings = ChunkingSettings(
            target_chars=200, max_chars=400, min_chars=0, overlap_chars=50
        )
        doc = make_document("word " * 100)  # 500 字符，无空行 → 单段
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) > 1
        assert all(chunk.char_count <= settings.max_chars for chunk in chunks)

    def test_max_chars_is_a_hard_limit(self) -> None:
        """max_chars 必须是**硬上限**：任何片段的长度都不得超过它。

        早先的实现对"只超了一点"的单句放行（容忍到 2 倍），
        结果是块可以膨胀到任意大小，送进 LLM 上下文时会直接撞上 token 上限。
        """
        settings = ChunkingSettings(
            target_chars=200, max_chars=400, min_chars=0, overlap_chars=0
        )
        doc = make_document("".join(f"这是第{i}个句子。" for i in range(100)))
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) > 1
        assert all(chunk.char_count <= settings.max_chars for chunk in chunks)

    def test_max_chars_splits_at_sentence_boundary(self) -> None:
        settings = ChunkingSettings(
            target_chars=200, max_chars=400, min_chars=0, overlap_chars=0
        )
        text = "".join(f"这是第{i}个句子。" for i in range(100))
        doc = make_document(text)
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) > 1
        # 按句切分时每个片段都应以句末标点收尾，不出现半句话
        assert all(chunk.text.endswith("。") for chunk in chunks)

    def test_overlap_present_between_split_chunks(self) -> None:
        """切分点两侧必须保留重叠，否则恰好跨刀口的关键句两边都检索不到。"""
        settings = ChunkingSettings(
            target_chars=200, max_chars=400, min_chars=0, overlap_chars=50
        )
        text = "".join(f"句子内容编号{i}。" for i in range(100))
        doc = make_document(text)
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) >= 2
        last_sentence = chunks[0].text.split("。")[-2] + "。"
        assert last_sentence in chunks[1].text

    def test_fragment_ids_are_deterministic(self) -> None:
        doc = make_document("1 A\n\nSome text here.\n\nMore text there.")
        first = chunk_document(doc, source_key="s1")
        second = chunk_document(doc, source_key="s1")
        assert [item.fragment_id for item in first] == [item.fragment_id for item in second]

    def test_fragment_ids_differ_across_sources(self) -> None:
        doc = make_document("Some text here.")
        assert chunk_document(doc, source_key="s1")[0].fragment_id != chunk_document(
            doc, source_key="s2"
        )[0].fragment_id

    def test_chunk_index_is_monotonic(self) -> None:
        doc = make_document("1 A\n\nText one.\n\nText two long enough to matter.")
        chunks = chunk_document(doc, source_key="s1")
        assert [item.chunk_index for item in chunks] == list(range(len(chunks)))

    def test_appendix_survives_chunking(self) -> None:
        """端到端：附录内容必须真的出现在片段里。"""
        doc = make_document(
            "Main body paragraph with substantial content here.\n\n"
            "References\n\n"
            "[1] Someone. A paper title. 2020.\n[2] Other. Another paper. 2021.\n\n"
            "8 Methods\n\n"
            "The appendix describes implementation parameters in detail."
        )
        chunks = chunk_document(doc, source_key="s1")
        joined = "\n".join(chunk.text for chunk in chunks)
        assert "appendix describes implementation" in joined
        assert "Another paper" not in joined

    def test_references_dropped_when_configured(self) -> None:
        doc = make_document(
            "Body paragraph with real content.\n\nReferences\n\n"
            "[1] Someone. A paper title. 2020.\n[2] Other. Another paper.\n[3] Third.\n"
        )
        kept = chunk_document(
            doc, source_key="s1", settings=ChunkingSettings(drop_references=False)
        )
        dropped = chunk_document(
            doc, source_key="s1", settings=ChunkingSettings(drop_references=True)
        )
        assert any("Another paper" in chunk.text for chunk in kept)
        assert not any("Another paper" in chunk.text for chunk in dropped)

    def test_min_chars_merges_small_chunks_within_section(self) -> None:
        settings = ChunkingSettings(
            target_chars=1000, max_chars=2000, min_chars=200, overlap_chars=0
        )
        doc = make_document("1 A\n\nTiny.\n\nAnother tiny bit.")
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) == 1

    def test_small_chunks_not_merged_across_sections(self) -> None:
        """跨章节合并会让引用位置失真——"结论"的句子被标成属于"方法"。"""
        settings = ChunkingSettings(
            target_chars=1000, max_chars=2000, min_chars=500, overlap_chars=0
        )
        doc = make_document("1 A\n\nShort one.\n\n2 B\n\nShort two.")
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) == 2
        assert chunks[0].section_path == ["1 A"]

    def test_char_count_matches_text(self) -> None:
        doc = make_document("Some reasonably long paragraph text for counting.")
        for chunk in chunk_document(doc, source_key="s1"):
            assert chunk.char_count == len(chunk.text)

    def test_multiline_paragraph_joined(self) -> None:
        doc = make_document("A paragraph that is\nsoft wrapped across lines.")
        chunks = chunk_document(doc, source_key="s1")
        assert chunks[0].text == "A paragraph that is soft wrapped across lines."

    def test_long_single_sentence_is_hard_split(self) -> None:
        """没有标点的超长文本（表格行、公式）必须硬切，否则会无限增长。"""
        settings = ChunkingSettings(
            target_chars=200, max_chars=400, min_chars=0, overlap_chars=0
        )
        doc = make_document("x" * 1000)
        chunks = chunk_document(doc, source_key="s1", settings=settings)
        assert len(chunks) >= 3
        assert all(chunk.char_count <= settings.max_chars for chunk in chunks)

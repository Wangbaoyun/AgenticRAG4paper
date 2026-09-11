"""测试领域模型：文献标识、片段、证据、引用与会话。

领域模型是全部组件之间的公共语言，其不变式一旦被破坏，错误会以"引用错位""索引重建"
这类难以定位的形式在上层浮现。因此这里逐条钉住 SPEC §2–§3 的规定语义。
"""

from __future__ import annotations

import pytest
from scitrace.domain import (
    Answer,
    Evidence,
    Fragment,
    Session,
    SessionStatus,
    Source,
    SourcePatch,
    Usage,
    make_evidence_key,
    make_source_key,
    render_bibtex,
    render_inline_citation,
)
from scitrace.domain.citation import bibtex_entry_type, sanitize_reference_key
from scitrace.domain.fragment import ParsedDocument, ParsedPage, make_fragment_id
from scitrace.domain.source import normalize_doi


class TestDoiNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.5555/Example.2024.001", "10.5555/example.2024.001"),
            ("https://doi.org/10.1234/ABC", "10.1234/abc"),
            ("http://dx.doi.org/10.1234/ABC", "10.1234/abc"),
            ("doi:10.1234/ABC", "10.1234/abc"),
            ("DOI: 10.1234/ABC", "10.1234/abc"),
            ("  10.1234/abc  ", "10.1234/abc"),
        ],
    )
    def test_variants_collapse_to_one_form(self, raw: str, expected: str) -> None:
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "doi:"])
    def test_empty_returns_none(self, raw: str | None) -> None:
        assert normalize_doi(raw) is None


class TestSourceKey:
    def test_doi_variants_share_one_key(self) -> None:
        """同一篇论文的不同 DOI 写法与不同 PDF 副本必须归并为同一篇文献。"""
        key_a = make_source_key(doi="https://doi.org/10.1234/ABC", content_hash="aaaa")
        key_b = make_source_key(doi="10.1234/abc", content_hash="bbbb")
        assert key_a == key_b

    def test_without_doi_falls_back_to_content_hash(self) -> None:
        key_a = make_source_key(doi=None, content_hash="aaaa")
        key_b = make_source_key(doi=None, content_hash="bbbb")
        assert key_a != key_b

    def test_key_length_is_16_hex(self) -> None:
        key = make_source_key(doi=None, content_hash="aaaa")
        assert len(key) == 16
        assert all(char in "0123456789abcdef" for char in key)


class TestSourcePatch:
    def test_fill_gaps_does_not_overwrite(self) -> None:
        """核心合并语义：高优先级来源已提供的字段不得被低优先级来源覆盖。"""
        high = SourcePatch(title="High", year=2024)
        low = SourcePatch(title="Low", year=1999, venue="Nature")
        merged = high.fill_gaps_from(low)
        assert merged.title == "High"
        assert merged.year == 2024
        assert merged.venue == "Nature"

    def test_fill_gaps_is_pure(self) -> None:
        high = SourcePatch(title="High")
        high.fill_gaps_from(SourcePatch(venue="Nature"))
        assert high.venue is None

    def test_provided_fields_ignores_none(self) -> None:
        """None 表示"未提供"，不能被计为贡献——否则可观测性指标会失真。"""
        assert SourcePatch(title="x", year=None).provided_fields() == {"title"}

    def test_doi_is_normalized_on_construction(self) -> None:
        assert SourcePatch(doi="https://doi.org/10.1/AB").doi == "10.1/ab"

    def test_quality_tier_bounds(self) -> None:
        with pytest.raises(ValueError):
            SourcePatch(quality_tier=4)
        with pytest.raises(ValueError):
            SourcePatch(quality_tier=-1)


class TestSourceApplyPatch:
    def test_fills_missing_and_records_provider(self, source_zh: Source) -> None:
        enriched = source_zh.apply_patch(
            SourcePatch(doi="10.1234/x", year=2023, venue="计算机学报"),
            provider="crossref",
        )
        assert enriched.doi == "10.1234/x"
        assert enriched.year == 2023
        assert enriched.venue == "计算机学报"
        assert enriched.metadata_sources == ["crossref"]

    def test_does_not_overwrite_existing_values(self, source: Source) -> None:
        enriched = source.apply_patch(SourcePatch(title="Wrong", venue="Wrong"), provider="x")
        assert enriched.title == source.title
        assert enriched.venue == source.venue

    def test_does_not_mutate_original(self, source: Source) -> None:
        source.apply_patch(SourcePatch(abstract="new"), provider="x")
        assert source.abstract is None
        assert source.metadata_sources == []

    def test_provider_list_is_deduplicated_and_ordered(self, source_zh: Source) -> None:
        enriched = (
            source_zh.apply_patch(SourcePatch(doi="10.1/a"), provider="crossref")
            .apply_patch(SourcePatch(citation_count=7), provider="semantic_scholar")
            .apply_patch(SourcePatch(abstract="x"), provider="crossref")
        )
        assert enriched.metadata_sources == ["crossref", "semantic_scholar"]

    def test_empty_patch_is_a_noop(self, source: Source) -> None:
        assert source.apply_patch(SourcePatch(), provider="x") is source

    def test_retracted_only_ever_set_true(self, source: Source) -> None:
        """撤稿标记是单向的：False 补丁不能把已知撤稿"洗白"，

        否则一个不查撤稿库的 provider 就能覆盖掉已查实的撤稿状态。
        """
        flagged = source.apply_patch(SourcePatch(retracted=True), provider="retractionwatch")
        assert flagged.retracted is True

        cleaned = flagged.apply_patch(SourcePatch(retracted=False), provider="crossref")
        assert cleaned.retracted is True

    def test_metadata_complete_flag(self, source: Source, source_zh: Source) -> None:
        assert source.metadata_complete is True
        assert source_zh.metadata_complete is False

        completed = (
            source_zh.apply_patch(SourcePatch(doi="10.1/a"), provider="crossref")
            .apply_patch(SourcePatch(year=2023), provider="crossref")
        )
        assert completed.metadata_complete is True


class TestCitationStem:
    def test_latin_name_and_acronym(self, source: Source) -> None:
        assert source.citation_stem == "whitfield2024adaptive"

    def test_comma_form_surname(self) -> None:
        src = Source(
            key="k",
            content_hash="h",
            rel_path="p",
            title="Retrieval Augmented Generation",
            authors=["Whitfield, Dana R."],
            year=2024,
        )
        assert src.citation_stem == "whitfield2024retrieval"

    def test_missing_year_uses_nd_not_current_year(self, source_zh: Source) -> None:
        """必须用 'nd' 而非当前年份：否则同一文献在不同年份引用时键会变。"""
        assert "nd" in source_zh.citation_stem
        assert "2026" not in source_zh.citation_stem

    def test_chinese_title_yields_chinese_token(self) -> None:
        src = Source(
            key="k",
            content_hash="h",
            rel_path="p",
            title="面向科研文献的证据可溯源问答方法研究",
            authors=["张三"],
            year=2023,
        )
        assert src.citation_stem == "张三2023面向科研"

    def test_no_author_falls_back_to_anon(self) -> None:
        src = Source(key="k", content_hash="h", rel_path="p", title="Some Study", year=2020)
        assert src.citation_stem.startswith("anon2020")

    def test_uppercase_acronym_preserved(self) -> None:
        src = Source(
            key="k",
            content_hash="h",
            rel_path="p",
            title="SPECTRA for Sequence Labeling",
            authors=["Ferraro"],
            year=2019,
        )
        assert src.citation_stem == "ferraro2019SPECTRA"

    def test_stopword_only_title_falls_back(self) -> None:
        src = Source(key="k", content_hash="h", rel_path="p", title="The Of And", year=2020)
        assert src.citation_stem == "anon2020untitled"


class TestFragment:
    def test_page_label_ranges(self) -> None:
        base = {
            "fragment_id": "f",
            "source_key": "s",
            "text": "x",
            "chunk_index": 0,
            "char_count": 1,
        }
        assert Fragment(**base, page_start=3, page_end=3).page_label == "page 3"
        assert Fragment(**base, page_start=3, page_end=4).page_label == "pages 3-4"
        assert Fragment(**base, page_start=3).page_label == "page 3"
        assert Fragment(**base).page_label == ""

    def test_section_label(self, fragment: Fragment) -> None:
        assert fragment.section_label == "4 Results"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="不能为空"):
            Fragment(
                fragment_id="f",
                source_key="s",
                text="   ",
                chunk_index=0,
                char_count=0,
            )

    def test_fragment_id_is_deterministic_and_positional(self) -> None:
        args = {"source_key": "s", "section_path": ["1 Intro"], "chunk_index": 0}
        assert make_fragment_id(**args) == make_fragment_id(**args)
        assert make_fragment_id(**args) != make_fragment_id(
            source_key="s", section_path=["1 Intro"], chunk_index=1
        )
        assert make_fragment_id(**args) != make_fragment_id(
            source_key="s", section_path=["2 Method"], chunk_index=0
        )

    def test_parsed_document_helpers(self) -> None:
        doc = ParsedDocument(
            pages=[ParsedPage(page_number=1, text="a"), ParsedPage(page_number=2, text="b")],
            parser="pypdf",
        )
        assert doc.n_pages == 2
        assert doc.full_text == "a\n\nb"


class TestEvidence:
    def test_key_is_deterministic_and_prefixed(self) -> None:
        key = make_evidence_key("s1", "f1")
        assert key == make_evidence_key("s1", "f1")
        assert key.startswith("ev-")
        assert len(key) == 3 + 8

    def test_key_differs_by_fragment_and_source(self) -> None:
        assert make_evidence_key("s1", "f1") != make_evidence_key("s1", "f2")
        assert make_evidence_key("s1", "f1") != make_evidence_key("s2", "f1")

    def test_relevance_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            Evidence(key="ev-1", source_key="s", fragment_id="f", summary="x", relevance=11)
        with pytest.raises(ValueError):
            Evidence(key="ev-1", source_key="s", fragment_id="f", summary="x", relevance=-1)

    def test_is_relevant(self) -> None:
        def make(score: int) -> Evidence:
            return Evidence(
                key="ev-1", source_key="s", fragment_id="f", summary="x", relevance=score
            )

        assert make(0).is_relevant is False
        assert make(1).is_relevant is True

    def test_empty_summary_rejected(self) -> None:
        with pytest.raises(ValueError):
            Evidence(key="ev-1", source_key="s", fragment_id="f", summary="  ", relevance=5)


class TestCitationRendering:
    def test_inline_with_and_without_pages(self) -> None:
        assert render_inline_citation("a2024t", "pages 3-4") == "(a2024t pages 3-4)"
        assert render_inline_citation("a2024t", "") == "(a2024t)"

    def test_inline_empty_stem_is_marked_unknown(self) -> None:
        """宁可显示 unknown，也不产生一个看起来像真的空引用。"""
        assert render_inline_citation("", "page 1") == "(unknown page 1)"

    def test_entry_type_selection(self, source: Source, source_zh: Source) -> None:
        assert bibtex_entry_type(source) == "article"
        assert bibtex_entry_type(source_zh) == "misc"

    def test_sanitize_keeps_cjk_but_drops_illegal(self) -> None:
        assert sanitize_reference_key("张三2023面向科研") == "张三2023面向科研"
        assert sanitize_reference_key("a b,c{d}") == "abcd"
        assert sanitize_reference_key("///") == "///"
        assert sanitize_reference_key(" , ") == "ref"

    def test_bibtex_is_parseable_by_pybtex(self, source: Source, source_zh: Source) -> None:
        """BibTeX 正确性由外部实现反向校验，而不是自说自话。"""
        from pybtex.database import parse_string

        for item in (source, source_zh):
            bib = render_bibtex(item)
            database = parse_string(bib, bib_format="bibtex")
            assert len(database.entries) == 1

            entry = next(iter(database.entries.values()))
            assert entry.type == bibtex_entry_type(item)
            assert str(entry.fields["title"]) == item.title
            if item.year:
                assert entry.fields["year"] == str(item.year)

    def test_bibtex_omits_empty_fields(self, source_zh: Source) -> None:
        """不输出空字段：空字段会被解析成"已知为空"，比缺失更难排查。"""
        bib = render_bibtex(source_zh)
        assert "doi = {}" not in bib
        assert "journal = {}" not in bib
        assert "year" not in bib

    def test_bibtex_escapes_specials(self) -> None:
        src = Source(
            key="k",
            content_hash="h",
            rel_path="p",
            title="Attention & Memory: 100% Recall_Test",
            authors=["A"],
            year=2020,
        )
        bib = render_bibtex(src)
        assert r"\&" in bib
        assert r"\%" in bib
        assert r"\_" in bib

    def test_retracted_is_surfaced(self, source: Source) -> None:
        """撤稿必须显眼——把撤稿结论当有效证据是最严重的失效模式。"""
        flagged = source.apply_patch(SourcePatch(retracted=True), provider="rw")
        assert "RETRACTED" in render_bibtex(flagged)

    def test_custom_reference_key(self, source: Source) -> None:
        assert "mykey," in render_bibtex(source, reference_key="mykey")


class TestUsage:
    def test_merge_sums_all_fields(self) -> None:
        a = Usage(prompt_tokens=10, completion_tokens=5, llm_calls=1, parse_failures=2)
        b = Usage(prompt_tokens=1, completion_tokens=2, llm_calls=3, cache_hits=4)
        merged = a.merge(b)
        assert merged.prompt_tokens == 11
        assert merged.completion_tokens == 7
        assert merged.llm_calls == 4
        assert merged.parse_failures == 2
        assert merged.cache_hits == 4

    def test_merge_is_pure(self) -> None:
        a = Usage(prompt_tokens=10)
        a.merge(Usage(prompt_tokens=5))
        assert a.prompt_tokens == 10

    def test_total_tokens(self) -> None:
        assert Usage(prompt_tokens=3, completion_tokens=4).total_tokens == 7


class TestSession:
    def test_new_is_deterministic(self) -> None:
        """同问题同索引 → 同 session_id，回归测试与消融实验才能稳定比对。"""
        a = Session.new(question="Q?", fingerprint="fp1")
        b = Session.new(question="Q?", fingerprint="fp1")
        assert a.session_id == b.session_id

    def test_new_differs_by_fingerprint(self) -> None:
        a = Session.new(question="Q?", fingerprint="fp1")
        b = Session.new(question="Q?", fingerprint="fp2")
        assert a.session_id != b.session_id

    def test_blank_question_rejected(self) -> None:
        with pytest.raises(ValueError):
            Session.new(question="   ", fingerprint="fp")

    def test_default_status_is_fail(self) -> None:
        """默认值必须是最悲观的终态：忘记设置状态时应暴露问题而非伪装成功。"""
        assert Session.new(question="q", fingerprint="fp").status is SessionStatus.FAIL

    def test_get_evidence(self) -> None:
        session = Session.new(question="q", fingerprint="fp")
        item = Evidence(key="ev-1", source_key="s", fragment_id="f", summary="x", relevance=5)
        session.evidence.append(item)
        assert session.get_evidence("ev-1") is item
        assert session.get_evidence("ev-nope") is None

    def test_json_roundtrip_preserves_content(self, source: Source) -> None:
        session = Session.new(question="中文问题", fingerprint="fp")
        session.status = SessionStatus.SUCCESS
        session.answer = Answer(text="答案", raw_text="答案")
        session.usage = Usage(prompt_tokens=1)

        restored = Session.from_json(session.to_json())
        assert restored == session
        assert "中文问题" in session.to_json()

    def test_answer_has_citations(self) -> None:
        assert Answer(text="x").has_citations is False

"""测试解析器层：编码探测、分页、注册表与错误语义。

解析器是整条链路里错误最集中的一环（PDF 千奇百怪、中文编码五花八门），
而且它的失败方式很隐蔽：用错编码读中文**不会抛异常**，只会得到乱码然后
被安静地索引进库，最终表现为"检索什么都搜不到"。因此本文件的重点是失败路径。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_fixture import write_blank_pdf, write_text_pdf
from scitrace.adapters.parsers import (
    available_parsers,
    get_parser,
    register_parser,
    select_parser,
)
from scitrace.adapters.parsers.plaintext import (
    PlainTextParser,
    decode_text_bytes,
    describe_encoding,
    document_language_ratio,
)
from scitrace.adapters.parsers.pypdf_parser import PyPDFParser
from scitrace.ports import DocumentParser, ParseError


class TestDecodeTextBytes:
    def test_utf8(self) -> None:
        assert decode_text_bytes("证据可溯源".encode()) == "证据可溯源"

    def test_utf8_with_bom(self) -> None:
        """带 BOM 的 UTF-8 必须去掉 BOM，否则首个字符会带上不可见字符，
        使该文档的第一个 bigram 与查询永远对不上。"""
        assert decode_text_bytes("\ufeff证据可溯源".encode("utf-8")) == "证据可溯源"

    def test_gb18030(self) -> None:
        raw = "中文科研文献问答".encode("gb18030")
        assert decode_text_bytes(raw) == "中文科研文献问答"

    def test_big5(self) -> None:
        raw = "中文文獻檢索問答系統".encode("big5")
        assert decode_text_bytes(raw) == "中文文獻檢索問答系統"

    def test_utf16(self) -> None:
        raw = "中文 UTF-16 文本".encode("utf-16")
        assert "UTF-16" in decode_text_bytes(raw)

    def test_ascii_is_utf8(self) -> None:
        assert decode_text_bytes(b"plain ascii") == "plain ascii"

    def test_malformed_bytes_do_not_raise(self) -> None:
        """SPEC §3.10：编码异常字符替换后继续，不抛异常。"""
        result = decode_text_bytes(b"\xff\xfe\x00\x80broken")
        assert isinstance(result, str)

    def test_empty(self) -> None:
        assert decode_text_bytes(b"") == ""

    def test_describe_encoding_reports_source(self, tmp_path: Path) -> None:
        target = tmp_path / "zh.txt"
        target.write_bytes("中文".encode("gb18030"))
        assert describe_encoding(target) in {"gb18030", "utf-8"}


class TestPlainTextParser:
    def test_supports(self) -> None:
        parser = PlainTextParser()
        assert parser.supports(Path("a.txt"))
        assert parser.supports(Path("a.md"))
        assert parser.supports(Path("A.MD"))
        assert not parser.supports(Path("a.pdf"))

    async def test_parse_plain_file(self, tmp_path: Path) -> None:
        target = tmp_path / "notes.md"
        target.write_text("## Method\n\nSome text here.", encoding="utf-8")

        document = await PlainTextParser().parse(target)
        assert document.parser == "plaintext"
        assert document.n_pages == 1
        assert "Some text here." in document.full_text
        assert document.hints["title"] == "notes"

    async def test_form_feed_creates_pages(self, tmp_path: Path) -> None:
        target = tmp_path / "paged.txt"
        target.write_text("page one\fpage two\fpage three", encoding="utf-8")

        document = await PlainTextParser().parse(target)
        assert document.n_pages == 3
        assert document.pages[1].page_number == 2
        assert document.pages[1].text == "page two"

    async def test_chinese_file_read_correctly(self, tmp_path: Path) -> None:
        target = tmp_path / "cn.txt"
        target.write_bytes("本文提出一种跨模态对齐方法。".encode("gb18030"))

        document = await PlainTextParser().parse(target)
        assert "跨模态对齐" in document.full_text

    async def test_missing_file_raises_parse_error(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match="无法读取"):
            await PlainTextParser().parse(tmp_path / "nope.txt")

    async def test_citation_key_hint_detected(self, tmp_path: Path) -> None:
        target = tmp_path / "Whitfield2024Adaptive.md"
        target.write_text("body", encoding="utf-8")
        document = await PlainTextParser().parse(target)
        assert document.hints["citation_key"] == "Whitfield2024Adaptive"

    async def test_language_ratio(self, tmp_path: Path) -> None:
        chinese = tmp_path / "cn.txt"
        chinese.write_text("中文科研文献问答系统", encoding="utf-8")
        english = tmp_path / "en.txt"
        english.write_text("research paper question answering", encoding="utf-8")

        assert document_language_ratio(await PlainTextParser().parse(chinese)) > 0.9
        assert document_language_ratio(await PlainTextParser().parse(english)) == 0.0


class TestPyPDFParser:
    def test_supports(self) -> None:
        parser = PyPDFParser()
        assert parser.supports(Path("a.pdf"))
        assert parser.supports(Path("A.PDF"))
        assert not parser.supports(Path("a.txt"))

    async def test_parse_generated_pdf(self, tmp_path: Path) -> None:
        target = write_text_pdf(tmp_path / "paper.pdf", ["Abstract", "We study retrieval."])

        document = await PyPDFParser().parse(target)
        assert document.parser == "pypdf"
        assert document.n_pages == 1
        assert "retrieval" in document.full_text
        assert document.hints["filename"] == "paper.pdf"

    async def test_blank_pdf_reports_scanned_document(self, tmp_path: Path) -> None:
        """无文本层必须**明确报错**而不是安静地索引一份空文档。

        静默跳过会让用户永远不知道自己的语料里有扫描件——检索不到东西时
        也无从排查。错误信息要直接指出原因与出路。
        """
        target = write_blank_pdf(tmp_path / "scan.pdf")
        with pytest.raises(ParseError, match="扫描件"):
            await PyPDFParser().parse(target)

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match="文件不存在"):
            await PyPDFParser().parse(tmp_path / "nope.pdf")

    async def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "broken.pdf"
        target.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
        with pytest.raises(ParseError, match="无法打开"):
            await PyPDFParser().parse(target)

    async def test_encrypted_pdf_raises(self, tmp_path: Path) -> None:
        from pypdf import PdfWriter

        target = tmp_path / "locked.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret-password")
        with target.open("wb") as handle:
            writer.write(handle)

        with pytest.raises(ParseError, match="加密"):
            await PyPDFParser().parse(target)


class TestParserRegistry:
    def test_available_parsers(self) -> None:
        assert set(available_parsers()) >= {"plaintext", "pypdf"}

    def test_get_parser_returns_protocol_instance(self) -> None:
        assert isinstance(get_parser("pypdf"), DocumentParser)
        assert isinstance(get_parser("plaintext"), DocumentParser)

    def test_unknown_name_lists_alternatives(self) -> None:
        """配置项写错一个字母却只得到"解析器不存在"，用户无从查起。"""
        with pytest.raises(ValueError, match="可用解析器"):
            get_parser("pymupdf-does-not-exist")

    def test_select_by_suffix(self) -> None:
        assert isinstance(select_parser(Path("a.pdf")), PyPDFParser)
        assert isinstance(select_parser(Path("a.txt")), PlainTextParser)

    def test_preferred_parser_is_honoured(self) -> None:
        assert isinstance(select_parser(Path("a.txt"), preferred="plaintext"), PlainTextParser)

    def test_preferred_parser_that_cannot_handle_file_raises(self) -> None:
        """刻意**不**静默回退：那会让"配置了 pymupdf 实际用了 pypdf"永远不被发现，
        而两者的输出差异会直接改变索引内容。"""
        with pytest.raises(ValueError, match="不支持"):
            select_parser(Path("a.txt"), preferred="pypdf")

    def test_unsupported_suffix_raises(self) -> None:
        with pytest.raises(ValueError, match="没有解析器"):
            select_parser(Path("a.docx"))

    def test_register_custom_parser(self) -> None:
        class DummyParser:
            name = "dummy-test"

            def supports(self, path: Path) -> bool:
                return path.suffix == ".dummy"

            async def parse(self, path: Path):  # pragma: no cover - 本测试不调用
                raise NotImplementedError

        try:
            register_parser(DummyParser.name, DummyParser)
            assert "dummy-test" in available_parsers()
            assert isinstance(select_parser(Path("x.dummy")), DummyParser)
        finally:
            from scitrace.adapters.parsers import _REGISTRY

            _REGISTRY.pop("dummy-test", None)

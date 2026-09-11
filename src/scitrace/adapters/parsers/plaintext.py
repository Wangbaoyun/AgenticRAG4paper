"""纯文本解析器（``.txt`` / ``.md`` / ``.markdown``）。

看似平凡，但中文场景下有一个必须处理好的问题：**编码**。
中文纯文本在真实语料里可能是 UTF-8、GB18030、Big5 或 UTF-16（带 BOM），
而用错误的编码读出来不会抛异常——只会得到一堆乱码，然后被安静地索引进库，
最后表现为"检索什么都搜不到"。因此这里的编码探测是**按顺序试解并用启发式校验**，
而不是简单地 ``errors="ignore"``。
"""

from __future__ import annotations

from pathlib import Path

import anyio

from scitrace.domain import ParsedDocument, ParsedPage
from scitrace.ports import ParseError
from scitrace.util.tokenize_zh import cjk_ratio

__all__ = ["PlainTextParser", "decode_text_bytes"]


#: 尝试的编码顺序。UTF-8 优先（现代语料主流），其次是中文常见编码。
#: ``gb18030`` 排在 ``gbk`` 前：它是 GBK 的超集，能覆盖更多生僻字，且对
#: GBK 文本同样能正确解码。
_CANDIDATE_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16")

#: 页分隔符。纯文本没有页概念，用换页符（form feed）作为显式分页信号。
_PAGE_BREAK = "\f"


def _looks_like_garbled(text: str) -> bool:
    """判断解码结果是否像乱码。

    启发式：替换字符 U+FFFD 占比过高，或大量字符落在 Unicode 私用区/未分配区。
    用错编码解中文的典型症状就是这两种。
    """
    if not text:
        return False
    sample = text[:4000]
    suspicious = sum(
        1
        for char in sample
        if char == "\ufffd" or 0xE000 <= ord(char) <= 0xF8FF or ord(char) == 0
    )
    return suspicious / len(sample) > 0.02


def decode_text_bytes(data: bytes) -> str:
    """把字节串解码为文本，自动处理常见中文编码。

    策略：按 :data:`_CANDIDATE_ENCODINGS` 顺序严格解码，取第一个**解码成功且不像乱码**
    的结果。全部失败时用 UTF-8 带替换字符兜底——宁可丢字符也不让整个文件摄入失败，
    因为 SPEC §3.10 要求"编码异常字符替换后继续，不抛异常"。

    Args:
        data: 文件原始字节。

    Returns:
        解码后的文本。
    """
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            text = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if not _looks_like_garbled(text):
            return text
    return data.decode("utf-8", errors="replace")


class PlainTextParser:
    """把纯文本 / Markdown 文件读成 :class:`ParsedDocument`。

    分页规则：以换页符 ``\\f`` 分页；没有换页符时整份文件视为一页
    （页码在引用中仍然有意义——它给出"在这份文档的第 1 页"这一确定位置，
    总好过完全缺失位置信息）。
    """

    #: 参与索引指纹的解析器名（SPEC §5.3）。
    name = "plaintext"

    SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".text"})

    def supports(self, path: Path) -> bool:
        """是否支持该文件（按后缀判断，大小写不敏感）。"""
        return path.suffix.lower() in self.SUFFIXES

    async def parse(self, path: Path) -> ParsedDocument:
        """读取并解析文件。

        Raises:
            ParseError: 文件不存在、不是文件，或无法读取。
        """
        return await anyio.to_thread.run_sync(self._parse_sync, Path(path))

    def _parse_sync(self, path: Path) -> ParsedDocument:
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ParseError(f"无法读取文本文件 {path}：{error}") from error

        text = decode_text_bytes(data)
        raw_pages = text.split(_PAGE_BREAK) if _PAGE_BREAK in text else [text]
        pages = [
            ParsedPage(page_number=index, text=content)
            for index, content in enumerate(raw_pages, start=1)
        ]

        # 同 PyPDFParser：文件名只作兜底，不冒充标题线索
        hints: dict[str, str] = {"fallback_title": path.stem}
        if _stem_looks_like_citation_key(path.stem):
            hints["citation_key"] = path.stem

        return ParsedDocument(pages=pages, hints=hints, parser=self.name)


def _stem_looks_like_citation_key(stem: str) -> bool:
    """文件名是否像 ``Author2024Keyword`` 形式的引用键。

    学术 PDF 的常见命名法。识别出来可作为元数据补全的起点，
    但仅作为**线索**记录在 ``hints`` 中，不直接当作书目数据——
    文件名不可信，用它覆盖真实元数据会造成难以察觉的错误。
    """
    if len(stem) < 8:
        return False
    has_digit = any(char.isdigit() for char in stem)
    has_alpha = any(char.isalpha() for char in stem)
    return has_digit and has_alpha and " " not in stem


def describe_encoding(path: Path) -> str:
    """返回文件实际使用的编码名，供排障使用。"""
    data = Path(path).read_bytes()
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            text = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if not _looks_like_garbled(text):
            return encoding
    return "utf-8 (replace)"


def document_language_ratio(document: ParsedDocument) -> float:
    """文档的 CJK 字符占比，供上层判断语料主导语言。"""
    return cjk_ratio(document.full_text)

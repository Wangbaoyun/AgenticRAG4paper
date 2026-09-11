"""引用渲染：把证据键变成可读的文内引用与 BibTeX 条目。

本模块负责"引用"这一核心承诺的**最后一公里**：答案里出现的每个键，
必须能被解析成一条人类可读、且能唯一定位文献的位置短语；参考文献列表必须是
合法 BibTeX。

实现上刻意**不依赖 pybtex**：领域层保持纯 stdlib（除 pydantic），
BibTeX 的正确性由测试用 pybtex 反向校验（见 ``tests/test_citation.py``），
而不是在运行时引入一个只为拼字符串而存在的重型依赖。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from scitrace.domain.source import Source, SourceKey

__all__ = [
    "Citation",
    "bibtex_entry_type",
    "render_bibtex",
    "render_inline_citation",
    "sanitize_reference_key",
]

_BIBTEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

#: BibTeX 条目键中**确实非法**的字符（逗号、空白、括号、引号、花括号、反斜杠、
#: 波浪号、井号、百分号）。刻意**不**限制为 ASCII：
#: 中文文献的条目键保留汉字是可读性的关键（`张三2024面向科研` 远比 `ref001` 有用），
#: 而 pybtex/biblatex 均能正确处理 UTF-8 键。这是 SPEC §9 增量 ③ 的一部分。
_REFKEY_UNSAFE_RE = re.compile(r"[\s,(){}\[\]\"'\\~#%&=]")


def sanitize_reference_key(key: str) -> str:
    """把引用词干净化为合法的 BibTeX 条目键。

    剥离 BibTeX 明确非法的字符（空白、逗号、括号、引号、反斜杠等），
    但**保留中日韩字符**以便中文文献的引用键可读。

    若净化后为空，返回 ``"ref"`` 以保证键非空——一个空键会让整条参考文献
    无法被引用，属于静默失效。
    """
    cleaned = _REFKEY_UNSAFE_RE.sub("", key).strip()
    return cleaned or "ref"


def bibtex_entry_type(source: Source) -> str:
    """选择 BibTeX 条目类型。

    有期刊/会议名 → ``@article``；否则按预印本处理为 ``@misc``。
    刻意不猜测 ``@inproceedings``：无法可靠区分会议论文与期刊论文时，
    ``@article`` 是更保守的选择（多数 BibTeX 样式都能正确排版）。
    """
    return "article" if source.venue else "misc"


def render_inline_citation(stem: str, page_label: str = "") -> str:
    """渲染文内引用，如 ``(skarlinski2024language pages 3-4)``。

    Args:
        stem: 引用词干（``Source.citation_stem``）。
        page_label: 位置短语，如 ``"pages 3-4"``；为空时只渲染词干。

    Returns:
        含圆括号的引用字符串。词干为空时返回 ``"(unknown)"``——
        宁可显示"未知"，也不产生一个看起来像真的空引用。
    """
    label = stem.strip() or "unknown"
    position = page_label.strip()
    return f"({label} {position})" if position else f"({label})"


def _escape(value: str) -> str:
    """转义 BibTeX 特殊字符。"""
    return "".join(_BIBTEX_SPECIALS.get(char, char) for char in value)


def render_bibtex(source: Source, *, reference_key: str | None = None) -> str:
    """把文献渲染为一条 BibTeX 条目。

    字段缺失时**省略该字段**，不输出空字段（如 ``doi = {}``）——
    空字段会被 pybtex 解析成"已知为空"，在参考文献表中呈现为错误的空白，
    比缺失更难排查。

    Args:
        source: 文献实体。
        reference_key: 条目键；默认用 ``source.citation_stem`` 净化后的结果。

    Returns:
        多行 BibTeX 文本，可被 pybtex 解析。
    """
    key = sanitize_reference_key(reference_key or source.citation_stem)
    entry_type = bibtex_entry_type(source)

    fields: list[tuple[str, str]] = [("title", _escape(source.title))]
    if source.authors:
        fields.append(("author", _escape(" and ".join(source.authors))))
    if source.year is not None:
        fields.append(("year", str(source.year)))
    if source.venue:
        fields.append(("journal", _escape(source.venue)))
    if source.doi:
        fields.append(("doi", _escape(source.doi)))
    if source.oa_url:
        fields.append(("url", _escape(source.oa_url)))
    if entry_type == "misc":
        fields.append(("howpublished", "Preprint"))
    if source.retracted:
        # 撤稿信息必须显眼：一条已撤稿的结论被当作有效证据引用，
        # 是文献问答系统最严重的失效模式之一。
        fields.append(("note", "RETRACTED"))

    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}"


class Citation(BaseModel):
    """答案中一处引用的完整解析结果——用于 ``--json`` 输出与可回溯性校验。"""

    model_config = ConfigDict(extra="forbid")

    evidence_key: str
    source_key: SourceKey
    reference_key: str
    inline: str
    page_label: str = ""

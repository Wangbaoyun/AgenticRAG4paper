"""文本抽取辅助：从正文里找出可用作元数据查询起点的标识符。

元数据补全（SPEC §3.5）需要 DOI 或标题作为查询线索。标题能从 PDF 元数据/文件名拿到，
但**DOI 几乎只能从正文里找**——绝大多数出版社的 PDF 会把 DOI 印在第一页。
在校验和清理之前先把它捞出来，能让精确查询端点的命中率大幅提高
（有 DOI 时 Crossref/OpenAlex 是 O(1) 精确查找，没有时只能走模糊检索，
准确率和限流表现都差很多）。
"""

from __future__ import annotations

import re

__all__ = ["find_arxiv_id", "find_doi", "first_page_text"]

#: DOI 的形式定义：``10.`` + 4~9 位注册机构号 + ``/`` + 后缀。
#: 后缀允许到空白/引号/尖括号为止，并在匹配后剥离尾部标点。
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)

#: DOI 后缀常被句子标点"粘"上，需要从尾部剥离。
_TRAILING_PUNCTUATION = ".,;:)]}>\"'"

#: arXiv 编号：``arXiv:2409.13740`` 或 ``arXiv:2409.13740v2``。
_ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def find_doi(text: str) -> str | None:
    """从文本中找出第一个 DOI。

    Args:
        text: 待搜索文本（通常取第一页）。

    Returns:
        归一化前的 DOI 原始串（清理尾部标点后）；未找到返回 ``None``。

    Note:
        返回的是**未经归一化**的串，交由
        :func:`scitrace.domain.source.normalize_doi` 统一处理，
        避免同一件事在两处实现。
    """
    match = _DOI_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).rstrip(_TRAILING_PUNCTUATION)
    # 括号配对保护：DOI 后缀里合法地含括号（如 10.1002/(SICI)1099-...），
    # 而句子末尾的右括号不是 DOI 的一部分。两者形式相同，
    # 用"数量是否配平"来区分。
    while candidate.count(")") > candidate.count("(") and candidate.endswith(")"):
        candidate = candidate[:-1]
    return candidate or None


def find_arxiv_id(text: str) -> str | None:
    """从文本中找出第一个 arXiv 编号（不含版本号）。"""
    match = _ARXIV_RE.search(text)
    return match.group(1) if match else None


def first_page_text(pages_text: list[str], *, limit: int = 4000) -> str:
    """取第一页的开头若干字符，用于查找 DOI / arXiv 编号。

    限制长度有两个理由：DOI 只印在第一页顶部；而超长文本会显著拖慢正则匹配
    （回溯风险），何况这份文本只用于查找标识符。
    """
    if not pages_text:
        return ""
    return pages_text[0][:limit]

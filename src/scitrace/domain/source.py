"""文献实体：本地可推导的标识 + 多源补全的书目元数据。

设计要点（SPEC §2.1、§3.5）：

- ``Source`` 把参考实现里"本地文档"与"带元数据的文档"两个概念**合并**为一个实体，
  并用 ``metadata_complete`` / ``metadata_sources`` 记录补全状态。合并不是简化，
  而是因为它们共享同一个稳定键 ``SourceKey``——分开表示会产生两者键不一致的风险。
- ``SourcePatch`` 是**稀疏**结构：``None`` 表示"该来源未提供此字段"，而不是"该字段为空"。
  这个区分是元数据合并语义的基础：没有它，后到的低优先级来源会用 ``None``
  覆盖先到的高优先级结果（SPEC §3.5 明确禁止覆盖已有非空字段）。
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scitrace.util.sanitize import SanitizedModel
from scitrace.util.hashing import normalize_text, sha256_hex

__all__ = ["Source", "SourceKey", "SourcePatch", "make_source_key", "normalize_doi"]

#: 文献的稳定键：16 位十六进制字符串。
SourceKey = str

_DOI_PREFIX_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", re.IGNORECASE)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ACRONYM_KEEP_RE = re.compile(r"^[A-Z]{2,}$")


def normalize_doi(doi: str | None) -> str | None:
    """把各种 DOI 书写形式归一化为裸 DOI。

    处理实际数据里高频出现的三种变体：URL 前缀（``https://doi.org/``）、
    ``doi:`` 前缀、大小写差异。归一化是必要的——否则同一篇论文的两种 DOI 写法
    会派生出两个 ``SourceKey``，导致重复入库。

    Args:
        doi: 原始 DOI 或 ``None``。

    Returns:
        归一化后的 DOI（小写、无前缀）；输入无效或为空时返回 ``None``。
    """
    if not doi:
        return None
    candidate = _DOI_PREFIX_RE.sub("", normalize_text(doi))
    candidate = re.sub(r"^doi:\s*", "", candidate, flags=re.IGNORECASE).strip().lower()
    return candidate or None


def make_source_key(*, doi: str | None, content_hash: str) -> SourceKey:
    """派生文献的稳定键。

    **优先用 DOI**：同一篇论文可能有多个 PDF 副本（预印本、会议版、出版社版），
    按内容哈希会得到不同的键而被当作多篇文献；按 DOI 则正确归并。
    无 DOI 时退化为内容哈希——至少保证同一文件重复入库是幂等的。

    Args:
        doi: 已归一化或未归一化的 DOI。
        content_hash: 文件内容哈希（``hash_file`` 的结果）。

    Returns:
        16 位十六进制键。
    """
    normalized = normalize_doi(doi)
    basis = f"doi:{normalized}" if normalized else f"hash:{content_hash}"
    return sha256_hex(basis, length=16)


class SourcePatch(SanitizedModel):
    """稀疏的书目元数据补丁。

    所有字段可空，``None`` 语义为"本来源未提供"，而非"该字段确实为空"。
    """

    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    is_oa: bool | None = None
    oa_url: str | None = None
    quality_tier: int | None = Field(default=None, ge=0, le=3)
    retracted: bool | None = None

    @field_validator("doi")
    @classmethod
    def _clean_doi(cls, value: str | None) -> str | None:
        return normalize_doi(value)

    @field_validator("title", "venue", "abstract")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = normalize_text(value)
        return cleaned or None

    @field_validator("authors")
    @classmethod
    def _clean_authors(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [normalize_text(author) for author in value]
        return [author for author in cleaned if author] or None

    def provided_fields(self) -> set[str]:
        """返回本补丁中**非 ``None``** 的字段名集合。

        供元数据合并逻辑判断"该来源贡献了什么"，也是可观测性指标
        （哪些 provider 真正起了作用）的数据来源。
        """
        return {name for name, value in self.model_dump().items() if value is not None}

    def fill_gaps_from(self, other: Self) -> Self:
        """用 ``other`` 填补本补丁中的空缺，返回新对象。

        **不许覆盖**：本对象已提供的字段一律保留。调用方按优先级从高到低依次
        ``patch = patch.fill_gaps_from(lower_priority_patch)``，
        即可得到"高优先级优先、且不被低优先级污染"的合并结果。

        Args:
            other: 优先级更低的补丁。

        Returns:
            合并后的新 ``SourcePatch``（本对象不被修改）。
        """
        merged = self.model_dump()
        for name, value in other.model_dump().items():
            if merged[name] is None and value is not None:
                merged[name] = value
        return type(self)(**merged)


class Source(SanitizedModel):
    """一篇文献：稳定标识、本地溯源信息与书目元数据。"""

    # ---- 标识与溯源 ----
    key: SourceKey
    content_hash: str
    rel_path: str = Field(description="相对于语料根目录的路径，用于报告与排障")

    # ---- 书目元数据（本地可推导 + 多源补全）----
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    abstract: str | None = None

    # ---- 质量与可信度信号 ----
    citation_count: int | None = None
    is_oa: bool | None = None
    oa_url: str | None = None
    quality_tier: int = Field(default=0, ge=0, le=3)
    retracted: bool = False

    # ---- 补全状态（可观测性）----
    metadata_sources: list[str] = Field(
        default_factory=list, description="按优先级顺序记录贡献过字段的来源名"
    )

    @field_validator("title")
    @classmethod
    def _require_title(cls, value: str) -> str:
        cleaned = normalize_text(value)
        if not cleaned:
            raise ValueError("Source.title 不能为空")
        return cleaned

    @property
    def metadata_complete(self) -> bool:
        """核心书目字段（title / authors / year / doi）是否齐全。

        刻意实现为**派生属性**而非存储字段：存储的布尔标志会与实际字段漂移——
        从清单反序列化、或调用方直接构造 ``Source`` 时，标志不会被重新计算，
        于是"元数据完整度"这项指标会静默失真。派生属性没有这个失效模式。
        """
        return (
            bool(self.title) and bool(self.authors) and self.year is not None and bool(self.doi)
        )

    def apply_patch(self, patch: SourcePatch, *, provider: str) -> Source:
        """把元数据补丁合入本文献，返回新对象。

        语义与 :meth:`SourcePatch.fill_gaps_from` 一致：**只填空缺，不覆盖**已有值。
        这样无论 provider 以什么顺序返回，结果都是确定的（SPEC §3.5 的字段级优先级
        由调用方通过返回顺序表达）。

        Args:
            patch: 待合入的稀疏补丁。
            provider: 来源名（如 ``"crossref"``），记入 ``metadata_sources``。

        Returns:
            合并后的新 ``Source``。
        """
        contributed = patch.provided_fields()
        if not contributed:
            return self

        data = self.model_dump()
        for name in contributed:
            current = data.get(name)
            # retracted 是布尔量，False 属于"有意义的已知值"，但初始值也是 False，
            # 无法区分"未查证"与"已查证为未撤稿"。因此只在补丁为 True 时采纳。
            if name == "retracted":
                if patch.retracted is True:
                    data["retracted"] = True
                continue
            if current is None or (isinstance(current, list | str) and not current):
                data[name] = getattr(patch, name)

        sources = list(self.metadata_sources)
        if provider not in sources:
            sources.append(provider)
        data["metadata_sources"] = sources
        return type(self)(**data)

    @property
    def citation_stem(self) -> str:
        """生成文内引用与 BibTeX 键共用的词干，形如 ``skarlinski2024language``。

        规则（本项目自定）：

        - 第一作者姓氏：取最后一个空格分隔片段并小写去非字母数字（处理
          ``"Smith, John"`` 与 ``"John Smith"`` 两种写法）；无作者时用 ``"anon"``；
        - 年份：缺失时用 ``"nd"``（no date），**不用当前年份**——那会让同一篇文献
          在不同年份引用时产生不同键，破坏确定性；
        - 标题词：取第一个"有信息量"的词。中文标题按字符取前 4 字（无空格可依），
          拉丁标题跳过停用词与长度 ≤ 3 的词。全大写缩写（如 ``"BERT"``）保留原形，
          便于人工识别。
        """
        first_author = "anon"
        if self.authors:
            surname = self.authors[0].split(",")[0] if "," in self.authors[0] else self.authors[0]
            parts = [part for part in surname.split() if part]
            raw = parts[-1] if parts else "anon"
            cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", raw).lower()
            first_author = cleaned or "anon"

        year = str(self.year) if self.year else "nd"
        return f"{first_author}{year}{_title_token(self.title)}"


#: 生成引用词干时跳过的英文停用词。
_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "toward",
        "towards",
        "using",
        "via",
        "with",
    }
)


def _title_token(title: str) -> str:
    """从标题中抽取用于引用词干的一个短标识。"""
    cjk_chars = [char for char in title if "\u4e00" <= char <= "\u9fff"]
    if len(cjk_chars) >= 2:
        return "".join(cjk_chars[:4])

    for word in re.split(r"[^0-9A-Za-z]+", title):
        if not word or word.lower() in _TITLE_STOPWORDS:
            continue
        if _ACRONYM_KEEP_RE.match(word):
            return word
        if len(word) > 3:
            return word.lower()
    return "untitled"

"""Crossref 元数据来源。

Crossref 是 DOI 的注册机构联盟，因此它的**权威性来自标识符**：只要 DOI 存在，
``/works/{doi}`` 返回的就是注册时登记的书目数据，是三个来源里最适合提供
title / authors / year / venue 的一个（SPEC §3.5 的字段级优先级也据此把
书目字段的首选给了它）。

## 这个来源的三个坑

1. **字段普遍是数组**：``title`` 是 ``["..."]``、``container-title`` 是 ``[]``。
   直接取下标会得到 ``list``（写进 pydantic 字段才报错），或者对空数组
   ``IndexError``——所以取首元素必须"取第一个非空字符串"而不是 ``[0]``。
2. **``issued.date-parts`` 是三层嵌套数组**（``[[2024, 5, 1]]``），
   且 ``issued`` 本身可能整个缺失。合法数据里还混着 ``[[0]]`` 这类占位值。
3. **``abstract`` 是 JATS 片段**，必须剥标签（见
   :func:`~scitrace.adapters.metadata.http_base.strip_markup`）。

## 标题检索为什么阈值卡在 0.8

``query.bibliographic`` 是**模糊**检索：给一个标题，Crossref 会返回一堆
"看起来相关"的论文，其中同主题不同工作的标题相似度经常在 0.5–0.7。
把阈值放低会让"补全"变成"改错"——用另一篇论文的年份与期刊覆盖本地的
可用信息。宁可不补，也不补错。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scitrace.adapters.metadata.http_base import (
    HttpMetadataProvider,
    as_int,
    as_mappings,
    as_text,
    as_year,
    dig,
    encode_doi_path,
    strip_markup,
)
from scitrace.domain import SourcePatch

__all__ = ["CrossrefProvider"]


class CrossrefProvider(HttpMetadataProvider):
    """Crossref REST API（``api.crossref.org``）。

    Args:
        mailto: 联系邮箱。Crossref 用它把请求归入 "polite pool"
            （限流额度更高、故障时更容易被人工处理）；缺省时不发送该参数。
        **kwargs: 透传给 :class:`~scitrace.adapters.metadata.http_base.HttpMetadataProvider`
            （``client`` / ``timeout_s`` / ``retry_policy`` / ``sleep``）。
    """

    SOURCE_NAME = "crossref"

    #: 两个端点共用前缀：DOI 端点是 ``{BASE}/{doi}``，检索端点是 ``{BASE}?query...``。
    BASE_URL = "https://api.crossref.org/works"

    #: 标题检索取前 N 个候选参与相似度比较。
    #:
    #: 5 是"够用且克制"的折中：真正对应的那篇在 ``query.bibliographic`` 下几乎
    #: 总在前 3，而取太多会让我们有机会把某个碰巧高分的错误候选选中。
    SEARCH_ROWS = 5

    def __init__(self, *, mailto: str | None = None, **kwargs: Any) -> None:
        super().__init__(default_params={"mailto": mailto} if mailto else None, **kwargs)

    # ------------------------------------------------------------ 子类接口 --

    async def _fetch_by_doi(self, doi: str) -> Mapping[str, Any] | None:
        """``/works/{doi}`` → 响应里 ``message`` 字段下的作品对象。"""
        payload = await self._request_json(f"{self.BASE_URL}/{encode_doi_path(doi)}")
        message = dig(payload, "message")
        return message if isinstance(message, Mapping) else None

    async def _search_by_title(self, title: str) -> Sequence[Mapping[str, Any]]:
        """``/works?query.bibliographic=`` → ``message.items`` 候选列表。"""
        payload = await self._request_json(
            self.BASE_URL,
            params={"query.bibliographic": title, "rows": str(self.SEARCH_ROWS)},
        )
        return as_mappings(dig(payload, "message", "items"))

    def _title_of(self, payload: Mapping[str, Any]) -> str | None:
        """作品标题：``title`` 是数组，取首个非空项并剥掉可能存在的 JATS 标记。"""
        return _clean(_first_text(payload.get("title")))

    def _patch_from_payload(self, payload: Mapping[str, Any]) -> SourcePatch:
        """把 Crossref 作品对象映射为稀疏补丁。

        与 SPEC §3.5 的字段对应关系：``title[0]`` / ``author[]`` /
        ``issued.date-parts[0][0]`` / ``container-title[0]`` / ``DOI`` /
        ``is-referenced-by-count`` / ``abstract``。
        """
        return SourcePatch(
            title=_clean(_first_text(payload.get("title"))),
            authors=_authors(payload),
            year=as_year(dig(payload, "issued", "date-parts", 0, 0)),
            doi=as_text(payload.get("DOI")),
            venue=_clean(_first_text(payload.get("container-title"))),
            abstract=_clean(payload.get("abstract")),
            citation_count=as_int(payload.get("is-referenced-by-count")),
        )


def _clean(value: Any) -> str | None:
    """取字符串并剥离 JATS/HTML 标记。

    Crossref 的 ``title`` / ``container-title`` / ``abstract`` 都可能是带标记的
    富文本（``<i>``、``<jats:p>``），用同一个函数处理可以避免"摘要干净了、
    标题却带着 ``<i>`` 进库"这种半吊子状态。
    """
    return strip_markup(as_text(value))


def _first_text(value: Any) -> str | None:
    """取数组里的第一个非空字符串；非数组输入返回 ``None``。

    不能写成 ``value[0]``：``container-title`` 在会议论文里常是 ``[]``
    （于是 ``IndexError`` 会让整个来源降级为"无贡献"，丢掉本来可用的
    title / authors），而 ``title`` 偶尔是 ``["", "..."]`` 这种带空首项的形状。
    """
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    for item in value:
        text = as_text(item)
        if text:
            return text
    return None


def _authors(payload: Mapping[str, Any]) -> list[str] | None:
    """把 ``author[]`` 拼成 ``"Given Family"`` 列表。

    ``given`` 可能缺失（中文、机构、只登记了姓氏的记录），此时只用 ``family``；
    部分条目既无 ``given`` 也无 ``family``，只有机构名 ``name``——若不处理，
    这类条目的作者会被整体丢弃。``None`` 而不是空列表：空列表在
    ``SourcePatch`` 里会被清洗成 ``None``，但显式返回 ``None`` 更能表达
    "本来源未提供作者"，与"作者确实为空"区分开。
    """
    names: list[str] = []
    for entry in as_mappings(payload.get("author")):
        given = as_text(entry.get("given"))
        family = as_text(entry.get("family"))
        if family and given:
            names.append(f"{given} {family}")
        elif family:
            names.append(family)
        else:
            organisation = as_text(entry.get("name"))
            if organisation:
                names.append(organisation)
    return names or None

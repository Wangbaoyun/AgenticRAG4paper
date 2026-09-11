"""Semantic Scholar Graph API 元数据来源。

在三个来源里，它的独有贡献是 **``citationCount``**（SPEC §3.5 把引用数
的首选给了它）与较完整的 ``venue``。它的 DOI 数据本身来自 Crossref，
所以书目字段的权威性不如 Crossref——这也正是合并层按字段分配优先级的原因。

## 限流是这个来源的主要工程约束

无 API key 时公开端点约 **1 req/s**，且额度由所有匿名调用者共享，
因此 429 是常态而非异常。这带来两个必须做对的地方：

1. **429 必须可重试**。``httpx`` 自己不会为非 2xx 抛错，而它的
   ``HTTPStatusError`` 又不带 ``status_code`` 属性，会被重试策略判为
   "不可重试"。基类因此把非 2xx 转成自带状态码的
   :class:`~scitrace.adapters.metadata.http_base.HttpStatusError`，
   让 429 落到 :data:`~scitrace.adapters.llm.retry.RETRYABLE_STATUS_CODES` 上。
2. **有 key 就带上**。``x-api-key`` 把请求归入个人额度，是从"经常 429"
   变成"基本可用"的唯一开关；没有配置时**不发送该请求头**，
   而不是发一个空值——空 ``x-api-key`` 会被判为鉴权失败（立刻失败、不重试）。

标题检索端点把候选放在 ``data`` 数组里，且 ``data`` 在无结果时可能缺失，
因此取候选同样走"缺失即空列表"的路径。
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
)
from scitrace.domain import SourcePatch

__all__ = ["SemanticScholarProvider"]


class SemanticScholarProvider(HttpMetadataProvider):
    """Semantic Scholar Graph API（``api.semanticscholar.org/graph/v1``）。

    Args:
        api_key: 可选的 API key（配置里是 ``SecretStr``，调用方需自行
            ``get_secret_value()`` 后传入）。缺省时不发送 ``x-api-key``。
        **kwargs: 透传给 :class:`~scitrace.adapters.metadata.http_base.HttpMetadataProvider`。
    """

    SOURCE_NAME = "semantic_scholar"

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

    #: 两个端点都**必须**显式声明字段：Graph API 默认只返回 ``paperId`` 与
    #: ``title``，不声明就会拿到一堆空字段，看起来像"来源没数据"。
    FIELDS = "title,year,venue,authors,externalIds,citationCount,abstract"

    #: 标题检索候选数（与 Crossref 保持同样的克制，理由见那边的说明）。
    SEARCH_LIMIT = 5

    def __init__(self, *, api_key: str | None = None, **kwargs: Any) -> None:
        headers = {"x-api-key": api_key} if api_key else None
        super().__init__(default_headers=headers, **kwargs)

    # ------------------------------------------------------------ 子类接口 --

    async def _fetch_by_doi(self, doi: str) -> Mapping[str, Any] | None:
        """``/paper/DOI:{doi}`` → 作品对象本身（**没有**外层包装字段）。"""
        payload = await self._request_json(
            f"{self.BASE_URL}/DOI:{encode_doi_path(doi)}", params={"fields": self.FIELDS}
        )
        return payload if isinstance(payload, Mapping) else None

    async def _search_by_title(self, title: str) -> Sequence[Mapping[str, Any]]:
        """``/paper/search`` → ``data`` 候选列表。"""
        payload = await self._request_json(
            f"{self.BASE_URL}/search",
            params={"query": title, "limit": str(self.SEARCH_LIMIT), "fields": self.FIELDS},
        )
        return as_mappings(dig(payload, "data"))

    def _title_of(self, payload: Mapping[str, Any]) -> str | None:
        return as_text(payload.get("title"))

    def _patch_from_payload(self, payload: Mapping[str, Any]) -> SourcePatch:
        """把作品对象映射为稀疏补丁。

        ``externalIds`` 可能是 ``null``（没有 DOI 的会议论文、预印本），
        因此 DOI 必须走 :func:`~scitrace.adapters.metadata.http_base.dig`
        而不是两层下标——那是本来源最容易踩的一处。
        """
        return SourcePatch(
            title=as_text(payload.get("title")),
            authors=_authors(payload),
            year=as_year(payload.get("year")),
            doi=as_text(dig(payload, "externalIds", "DOI")),
            venue=as_text(payload.get("venue")),
            abstract=as_text(payload.get("abstract")),
            citation_count=as_int(payload.get("citationCount")),
        )


def _authors(payload: Mapping[str, Any]) -> list[str] | None:
    """把 ``authors[]`` 取成姓名列表。

    条目形状是 ``{"authorId": "...", "name": "..."}``；``name`` 缺失
    （作者被合并或数据不全）时跳过该条，而不是写入 ``None`` 占位——
    书目字段里的空洞比缺作者更难排查。
    """
    names = [as_text(dig(entry, "name")) for entry in as_mappings(payload.get("authors"))]
    return [name for name in names if name] or None

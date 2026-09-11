"""OpenAlex 元数据来源。

OpenAlex 是三个来源里**唯一提供开放获取信息与场所类型**的一个，因此
SPEC §3.5 把 ``is_oa`` / ``oa_url`` / ``quality_tier`` 三个字段交给它。
它的数据来自 Crossref、PubMed、DOAJ、机构仓库等多方合并，覆盖面广，
代价是字段命名与嵌套形状都比前两个来源更"历史层积"。

## 一个字段两种名字

作品的标题同时存在 ``title`` 与 ``display_name``：老版本 API 只有
``display_name``（沿用实体通用的展示名字段），新版本引入 ``title`` 作为
正式别名，部分缓存与快照仍只带旧字段。两者都读、优先 ``title``，
否则一次 API 改版就会让所有标题变成"来源未提供"。

## quality_tier 是什么，不是什么

**这不是 JCR 分区，也不是任何期刊评价指标**，而是本项目自定的
**粗粒度可信度信号**（0–3），只依据 OpenAlex 的场所类型与 DOAJ 收录状态：

===========  ==========================================================
``type``     取值
===========  ==========================================================
缺失/空      0（有场所对象但没类型，按最低档处理；不猜成期刊）
repository   0（预印本、机构仓库：未经同行评审流程的版本居多）
conference   1
proceedings  1
journal      2；若 ``source.is_in_doaj is True`` 则为 3（DOAJ 只收录
             经同行评审的开放获取期刊，因此"期刊 + DOAJ"是可得的最强信号）
其他值       ``None``（``book``/``ebook platform``/``book series``/
             ``metadata`` 等无法映射到本项目的三档语义）
===========  ==========================================================

**无法判断时返回 ``None`` 而不是 0**：0 表示"已知是低可信档"，
``None`` 表示"本来源不知道"。二者在合并层的行为不同——``None`` 会让下一个
来源有机会回答，而错误的 0 会永久压低这篇文献的档位且无从察觉。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scitrace.adapters.metadata.http_base import (
    HttpMetadataProvider,
    as_bool,
    as_int,
    as_mappings,
    as_text,
    as_year,
    dig,
    encode_doi_path,
)
from scitrace.domain import SourcePatch
from scitrace.domain.source import normalize_doi

__all__ = ["OpenAlexProvider"]

#: ``primary_location.source.type`` → quality_tier 的已知取值。
#: 表里没有的类型一律 ``None``（不猜），理由见模块文档。
_TIER_BY_SOURCE_TYPE: Mapping[str, int] = {
    "repository": 0,
    "conference": 1,
    "proceedings": 1,
    "journal": 2,
}

#: 期刊 + DOAJ 收录 → 最高档。
_DOAJ_JOURNAL_TIER = 3


class OpenAlexProvider(HttpMetadataProvider):
    """OpenAlex API（``api.openalex.org``）。

    Args:
        mailto: 联系邮箱，OpenAlex 的 polite pool 参数；缺省时不发送。
        **kwargs: 透传给 :class:`~scitrace.adapters.metadata.http_base.HttpMetadataProvider`。
    """

    SOURCE_NAME = "openalex"

    BASE_URL = "https://api.openalex.org/works"

    #: 标题检索候选数。OpenAlex 的参数名是 ``per-page``（带连字符）。
    SEARCH_LIMIT = 5

    def __init__(self, *, mailto: str | None = None, **kwargs: Any) -> None:
        super().__init__(default_params={"mailto": mailto} if mailto else None, **kwargs)

    # ------------------------------------------------------------ 子类接口 --

    async def _fetch_by_doi(self, doi: str) -> Mapping[str, Any] | None:
        """``/works/doi:{doi}`` → **单个**作品对象（不是列表，也没有 ``message`` 包装）。"""
        payload = await self._request_json(f"{self.BASE_URL}/doi:{encode_doi_path(doi)}")
        return payload if isinstance(payload, Mapping) else None

    async def _search_by_title(self, title: str) -> Sequence[Mapping[str, Any]]:
        """``/works?search=`` → ``results`` 候选列表。"""
        payload = await self._request_json(
            self.BASE_URL,
            params={"search": title, "per-page": str(self.SEARCH_LIMIT)},
        )
        return as_mappings(dig(payload, "results"))

    def _title_of(self, payload: Mapping[str, Any]) -> str | None:
        return _title(payload)

    def _patch_from_payload(self, payload: Mapping[str, Any]) -> SourcePatch:
        """把作品对象映射为稀疏补丁。

        深层字段（``authorships[].author.display_name``、
        ``primary_location.source.display_name``）全部经
        :func:`~scitrace.adapters.metadata.http_base.dig` 读取：
        ``authorships`` 的每一项都可能缺 ``author``，
        ``primary_location`` 与它的 ``source`` 都可能是 ``null``——
        在预印本与会议论文里这是常态而非异常。
        """
        return SourcePatch(
            title=_title(payload),
            authors=_authors(payload),
            year=as_year(payload.get("publication_year")),
            # DOI 在 OpenAlex 里是 URL 形式（https://doi.org/10.x/y），
            # 必须归一化：否则同一个 DOI 会派生出两个 SourceKey。
            doi=normalize_doi(as_text(payload.get("doi"))),
            venue=as_text(dig(payload, "primary_location", "source", "display_name")),
            citation_count=as_int(payload.get("cited_by_count")),
            is_oa=as_bool(dig(payload, "open_access", "is_oa")),
            oa_url=as_text(dig(payload, "open_access", "oa_url")),
            quality_tier=_quality_tier(payload),
        )


def _title(payload: Mapping[str, Any]) -> str | None:
    """取标题：优先 ``title``，回退老版本的 ``display_name``。"""
    return as_text(payload.get("title")) or as_text(payload.get("display_name"))


def _authors(payload: Mapping[str, Any]) -> list[str] | None:
    """取 ``authorships[].author.display_name``。"""
    names = [
        as_text(dig(entry, "author", "display_name"))
        for entry in as_mappings(payload.get("authorships"))
    ]
    return [name for name in names if name] or None


def _quality_tier(payload: Mapping[str, Any]) -> int | None:
    """按场所类型与 DOAJ 收录状态映射 ``quality_tier``（0–3）。

    规则与取舍见模块文档。这里的判断顺序是刻意的：**先确认场所对象存在，
    再判断类型**。``primary_location`` 或 ``source`` 为 ``null`` 时我们
    连"这是期刊还是仓库"都不知道，属于"无法判断"，返回 ``None``；
    只有拿到了场所对象、但它的 ``type`` 为空时，才按最低档 0 处理。
    """
    source = dig(payload, "primary_location", "source")
    if not isinstance(source, Mapping):
        return None

    source_type = as_text(source.get("type"))
    if source_type is None:
        return 0

    tier = _TIER_BY_SOURCE_TYPE.get(source_type)
    if tier is None:
        return None
    if tier == _TIER_BY_SOURCE_TYPE["journal"] and source.get("is_in_doaj") is True:
        # 严格判 ``is True``：``is_in_doaj`` 缺失时为 ``None``，
        # 若用真值判断会把缺失误当作"已收录"，虚高一个档位。
        return _DOAJ_JOURNAL_TIER
    return tier

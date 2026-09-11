"""元数据合并：把多个来源的结果合成一份可信的书目补丁。

## 为什么是"字段级"优先级而非"来源级"

朴素做法是按来源排序、依次填补空缺（Crossref 的整份结果优先于 Semantic Scholar 的
整份结果）。这在本项目的数据上会立刻出错：**Crossref 不提供开放获取状态与被引数质量，
Semantic Scholar 的被引数比 OpenAlex 更及时，OpenAlex 的开放获取信息最全**。
按来源级排序意味着一旦 Crossref 命中，OA 链接与质量分级就永远补不上——
而它们恰恰是 Crossref 给不出的字段。

因此合并按**字段**分别选择来源（:data:`FIELD_PRIORITY`），这也正是 SPEC §3.5 的规定。

## 为什么置信度是第一排序键

用 DOI 精确查到的结果与用标题模糊搜到的结果必须区别对待：后者存在张冠李戴的可能
（同名论文、预印本与正式版、综述与原文）。若只按来源优先级排序，
一条"搜错了论文"的 Crossref 结果会压过一条 DOI 精确匹配的 OpenAlex 结果。
所以排序键的第一位是置信度，第二位才是来源优先级。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anyio

from scitrace.domain import SourcePatch
from scitrace.ports import Confidence, MetadataMatch, MetadataProvider

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_FIELD_PRIORITY", "MetadataResolverImpl"]

#: 置信度排序：数字越小越优先。
_CONFIDENCE_RANK: dict[Confidence, int] = {"doi": 0, "title": 1, "none": 2}

#: 字段级来源优先级（**SPEC §3.5 的落地**）。列表靠前者优先。
#:
#: 依据：
#: - 书目字段（标题/作者/年份/期刊/DOI）Crossref 最规范，因为它就是注册机构；
#: - 被引数 Semantic Scholar 更新更及时，OpenAlex 覆盖更广但滞后；
#: - 开放获取状态与链接 OpenAlex 最全（它聚合 Unpaywall）；
#: - 质量分级只有 OpenAlex 提供来源类型与 DOAJ 标记；
#: - 撤稿状态只信撤稿库（本地快照），任何书目来源都不作为撤稿的依据。
DEFAULT_FIELD_PRIORITY: dict[str, tuple[str, ...]] = {
    "title": ("crossref", "semantic_scholar", "openalex"),
    "authors": ("crossref", "semantic_scholar", "openalex"),
    "year": ("crossref", "semantic_scholar", "openalex"),
    "venue": ("crossref", "semantic_scholar", "openalex"),
    "doi": ("crossref", "semantic_scholar", "openalex"),
    "abstract": ("crossref", "semantic_scholar", "openalex"),
    "citation_count": ("semantic_scholar", "openalex", "crossref"),
    "is_oa": ("openalex", "unpaywall"),
    "oa_url": ("openalex", "unpaywall"),
    "quality_tier": ("openalex",),
    "retracted": ("retraction",),
}

#: 标题匹配的最低相似度。provider 内部已按同一阈值过滤，这里是**防御性复查**：
#: 组合层不能假设每个 provider 都正确实现了过滤——一个 provider 的疏漏
#: 会让错误标题进入书目数据，而错误标题会一路污染引用与参考文献。
DEFAULT_MIN_TITLE_SIMILARITY = 0.8


class MetadataResolverImpl:
    """满足 :class:`~scitrace.ports.MetadataResolver` 协议。

    并发调用全部 provider，各自独立超时与降级，最后按字段级优先级合并。
    """

    def __init__(
        self,
        providers: Sequence[MetadataProvider],
        *,
        timeout_s: float = 15.0,
        min_title_similarity: float = DEFAULT_MIN_TITLE_SIMILARITY,
        field_priority: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.providers = list(providers)
        self.timeout_s = timeout_s
        self.min_title_similarity = min_title_similarity
        self.field_priority = field_priority or DEFAULT_FIELD_PRIORITY

    @property
    def name(self) -> str:
        """组合器标识。

        会写入 ``Source.metadata_sources``——但它记录的是**配置了哪些来源**，
        而不是"这次实际是谁提供了数据"；后者由日志与
        :meth:`enrich` 的返回值共同体现。
        """
        return "+".join(provider.name for provider in self.providers) or "none"

    async def enrich(self, patch: SourcePatch) -> SourcePatch:
        """并发查询全部来源并按字段级优先级合并。

        Args:
            patch: 已知线索（通常来自 PDF 内嵌元数据、文件名与正文 DOI 提取）。

        Returns:
            合并后的补丁。**原始 ``patch`` 中已有的值不会被覆盖**——
            本地提取的标题即使不如数据库规范，也是"这篇文件自己的标题"，
            用外部标题覆盖它会让引用与用户看到的文件对不上。
            返回值仍会填补 ``patch`` 的空缺。
        """
        if not self.providers:
            return patch

        matches = await self._gather_matches(patch)
        if not matches:
            logger.debug("所有元数据来源均无结果（线索：doi=%s title=%r）", patch.doi, patch.title)
            return patch

        merged = self._merge(matches)
        result = patch.fill_gaps_from(merged)
        logger.debug(
            "元数据合并完成：来源 %s，新增字段 %s",
            sorted(matches),
            sorted(merged.provided_fields() - patch.provided_fields()),
        )
        return result

    async def _gather_matches(self, patch: SourcePatch) -> dict[str, MetadataMatch]:
        """并发调用全部 provider，收集有效匹配。

        每个 provider 有**独立**的超时预算：一个慢来源不能拖垮整次摄入。
        超出预算时该来源视为无贡献，其余来源的结果照常使用——
        这正是 SPEC §3.5 要求的"单源失败可降级"。
        """
        results: dict[str, MetadataMatch] = {}

        async def query(provider: MetadataProvider) -> None:
            match: MetadataMatch | None = None
            with anyio.move_on_after(self.timeout_s) as scope:
                try:
                    match = await provider.lookup(patch)
                except Exception as error:  # noqa: BLE001 - 组合层兜底，任何异常都不应中断
                    logger.warning("元数据来源 %s 抛出异常，已忽略：%s", provider.name, error)
                    return
            if scope.cancelled_caught:
                logger.warning("元数据来源 %s 超过 %.0f 秒未返回，已放弃", provider.name, self.timeout_s)
                return
            if match is None:
                return
            if not self._is_acceptable(match):
                return
            results[provider.name] = match

        async with anyio.create_task_group() as task_group:
            for provider in self.providers:
                task_group.start_soon(query, provider)
        return results

    def _is_acceptable(self, match: MetadataMatch) -> bool:
        """防御性复查：拒绝低置信度的标题匹配。

        DOI 匹配无相似度可言（标识符一致就是同一篇），不检查。
        """
        if match.confidence != "title":
            return True
        if match.score >= self.min_title_similarity:
            return True
        logger.warning(
            "来源 %s 的标题匹配相似度仅 %.2f（低于 %.2f），已丢弃：%r",
            match.provider,
            match.score,
            self.min_title_similarity,
            match.matched_title,
        )
        return False

    def _merge(self, matches: dict[str, MetadataMatch]) -> SourcePatch:
        """按字段逐个选择最优来源，产出合并补丁。"""
        chosen: dict[str, object] = {}
        for field in SourcePatch.model_fields:
            candidates = [
                (name, match)
                for name, match in matches.items()
                if getattr(match.patch, field, None) is not None
            ]
            if not candidates:
                continue

            # 撤稿是单向的：只有明确为 True 才采纳。
            # 一个没有撤稿库的来源不该有能力"洗白"已知撤稿。
            if field == "retracted":
                if any(getattr(match.patch, field) is True for _, match in candidates):
                    chosen[field] = True
                continue

            name, match = min(candidates, key=lambda item: self._rank(field, item))
            chosen[field] = getattr(match.patch, field)

        return SourcePatch(**chosen)  # type: ignore[arg-type]

    def _rank(self, field: str, candidate: tuple[str, MetadataMatch]) -> tuple[int, int, str]:
        """字段取值的排序键：置信度 → **该字段**的来源优先级 → 来源名。

        最后一个键是来源名，纯粹为了在置信度与优先级都相同时给出**确定的**结果——
        否则 ``min`` 会按字典插入顺序取值，而那是并发完成顺序决定的，
        会让同一次元数据补全在不同运行间产生不同结果。

        Args:
            field: 正在取值的字段名。优先级表是按字段组织的，
                因此必须由调用方显式传入——这正是"字段级优先级"的含义。
            candidate: ``(来源名, 匹配结果)``。
        """
        name, match = candidate
        priority = self.field_priority.get(field, ())
        # 不在该字段优先级表中的来源排在**所有**已知来源之后。
        # 这里必须用 ``len(priority)`` 而不是 ``len(self.providers)``：
        # 后者会与某个已知来源的下标碰撞（例如两个 provider 时 "openalex" 的下标
        # 恰好也是 2），碰撞后胜负就由来源名的字典序决定，而不是优先级。
        position = priority.index(name) if name in priority else len(priority)
        return (_CONFIDENCE_RANK[match.confidence], position, name)

    async def aclose(self) -> None:
        """释放全部 provider 的资源。单个 provider 关闭失败不影响其余。"""
        for provider in self.providers:
            try:
                await provider.aclose()
            except Exception as error:  # noqa: BLE001
                logger.warning("关闭元数据来源 %s 失败：%s", provider.name, error)

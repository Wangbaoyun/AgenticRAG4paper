"""测试元数据合并逻辑。

合并是整个元数据管线里唯一"有判断"的地方，也是最容易悄悄出错的地方：
错了不会抛异常，只会让某篇文献挂上别人的标题、或者明明有 OA 链接却补不上。
因此这里逐条钉住 SPEC §3.5 规定的合并语义。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scitrace.adapters.metadata.resolver import (
    DEFAULT_FIELD_PRIORITY,
    MetadataResolverImpl,
)
from scitrace.adapters.metadata.retraction import RetractionProvider
from scitrace.domain import SourcePatch
from scitrace.ports import MetadataMatch


class StubProvider:
    """可编程的元数据来源替身。"""

    def __init__(
        self,
        name: str,
        *,
        patch: SourcePatch | None = None,
        confidence: str = "doi",
        score: float = 1.0,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._patch = patch
        self._confidence = confidence
        self._score = score
        self._delay = delay
        self._error = error
        self.calls = 0
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    async def lookup(self, patch: SourcePatch) -> MetadataMatch | None:
        import anyio

        self.calls += 1
        if self._delay:
            await anyio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        if self._patch is None:
            return None
        return MetadataMatch(
            patch=self._patch,
            provider=self._name,
            confidence=self._confidence,  # type: ignore[arg-type]
            matched_title=self._patch.title,
            score=self._score,
        )

    async def aclose(self) -> None:
        self.closed = True


class TestBasics:
    async def test_no_providers_returns_patch_unchanged(self) -> None:
        resolver = MetadataResolverImpl([])
        patch = SourcePatch(title="Given")
        assert await resolver.enrich(patch) == patch

    async def test_name_lists_providers(self) -> None:
        resolver = MetadataResolverImpl(
            [StubProvider("crossref"), StubProvider("openalex")]
        )
        assert resolver.name == "crossref+openalex"

    async def test_empty_name_when_no_providers(self) -> None:
        assert MetadataResolverImpl([]).name == "none"

    async def test_single_provider_fills_gaps(self) -> None:
        resolver = MetadataResolverImpl(
            [StubProvider("crossref", patch=SourcePatch(year=2024, venue="Some Journal"))]
        )
        result = await resolver.enrich(SourcePatch(title="Given"))
        assert result.year == 2024
        assert result.venue == "Some Journal"
        assert result.title == "Given"

    async def test_existing_values_are_never_overwritten(self) -> None:
        """本地提取的标题是"这篇文件自己的标题"，外部标题覆盖它会让引用与文件对不上。"""
        resolver = MetadataResolverImpl(
            [StubProvider("crossref", patch=SourcePatch(title="Database Title"))]
        )
        result = await resolver.enrich(SourcePatch(title="Local Title"))
        assert result.title == "Local Title"

    async def test_no_matches_returns_patch(self) -> None:
        resolver = MetadataResolverImpl([StubProvider("crossref", patch=None)])
        patch = SourcePatch(title="Local")
        assert await resolver.enrich(patch) == patch


class TestFieldLevelPriority:
    """SPEC §3.5：优先级按**字段**取，不按来源全局取。

    这是本模块存在的核心理由——按来源级排序会让"Crossref 命中后
    OpenAlex 的 OA 链接永远补不上"，而 OA 链接恰恰是 Crossref 给不出的字段。
    """

    async def test_fields_are_sourced_independently(self) -> None:
        crossref = StubProvider("crossref", patch=SourcePatch(title="T", doi="10.1/x"))
        semantic_scholar = StubProvider("semantic_scholar", patch=SourcePatch(citation_count=42))
        openalex = StubProvider(
            "openalex", patch=SourcePatch(is_oa=True, oa_url="https://example.invalid/pdf")
        )
        resolver = MetadataResolverImpl([crossref, semantic_scholar, openalex])

        result = await resolver.enrich(SourcePatch(title="query"))
        assert result.doi == "10.1/x"  # 书目字段来自 crossref
        assert result.citation_count == 42  # 被引数来自 semantic_scholar
        assert result.is_oa is True  # OA 状态来自 openalex
        assert result.oa_url == "https://example.invalid/pdf"

    async def test_citation_count_prefers_semantic_scholar(self) -> None:
        openalex = StubProvider("openalex", patch=SourcePatch(citation_count=1))
        semantic_scholar = StubProvider("semantic_scholar", patch=SourcePatch(citation_count=99))
        resolver = MetadataResolverImpl([openalex, semantic_scholar])

        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.citation_count == 99, "顺序无关：字段优先级决定取值"

    async def test_bibliographic_fields_prefer_crossref(self) -> None:
        openalex = StubProvider("openalex", patch=SourcePatch(venue="OpenAlex Venue"))
        crossref = StubProvider("crossref", patch=SourcePatch(venue="Crossref Venue"))
        resolver = MetadataResolverImpl([openalex, crossref])

        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.venue == "Crossref Venue"

    async def test_unknown_provider_is_last_resort(self) -> None:
        """不在优先级表中的来源仍能补空缺，但排在已知来源之后。"""
        known = StubProvider("openalex", patch=SourcePatch(venue="Known"))
        unknown = StubProvider("mystery", patch=SourcePatch(venue="Unknown"))
        resolver = MetadataResolverImpl([unknown, known])

        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.venue == "Known"

    async def test_unknown_provider_still_fills_empty_field(self) -> None:
        unknown = StubProvider("mystery", patch=SourcePatch(abstract="only source"))
        resolver = MetadataResolverImpl([unknown])
        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.abstract == "only source"

    def test_priority_table_covers_every_patch_field(self) -> None:
        """优先级表必须覆盖 SourcePatch 的全部字段。

        漏掉一个字段不会报错，只会让该字段悄悄退化为"任意来源先到先得"。
        """
        assert set(DEFAULT_FIELD_PRIORITY) == set(SourcePatch.model_fields)


class TestConfidenceRanking:
    """置信度是排序键的第一位：一条"搜错了论文"的结果不该压过精确匹配。"""

    async def test_doi_confidence_beats_title_confidence(self) -> None:
        title_match = StubProvider(
            "crossref",
            patch=SourcePatch(venue="Wrong Paper Venue"),
            confidence="title",
            score=0.85,
        )
        doi_match = StubProvider(
            "openalex",
            patch=SourcePatch(venue="Right Paper Venue"),
            confidence="doi",
        )
        resolver = MetadataResolverImpl([title_match, doi_match])

        result = await resolver.enrich(SourcePatch(doi="10.1/x"))
        assert result.venue == "Right Paper Venue", "精确匹配必须压过模糊匹配"

    async def test_low_title_similarity_is_rejected(self) -> None:
        provider = StubProvider(
            "crossref",
            patch=SourcePatch(venue="Should Not Appear"),
            confidence="title",
            score=0.4,
        )
        resolver = MetadataResolverImpl([provider])
        result = await resolver.enrich(SourcePatch(title="query"))
        assert result.venue is None

    async def test_threshold_is_configurable(self) -> None:
        provider = StubProvider(
            "crossref", patch=SourcePatch(venue="V"), confidence="title", score=0.5
        )
        resolver = MetadataResolverImpl([provider], min_title_similarity=0.4)
        assert (await resolver.enrich(SourcePatch(title="q"))).venue == "V"

    async def test_doi_confidence_is_not_similarity_gated(self) -> None:
        """DOI 一致就是同一篇，不存在"相似度"一说，不该被阈值拦住。"""
        provider = StubProvider(
            "crossref",
            patch=SourcePatch(venue="V"),
            confidence="doi",
            score=0.0,
        )
        resolver = MetadataResolverImpl([provider])
        assert (await resolver.enrich(SourcePatch(doi="10.1/x"))).venue == "V"


class TestDegradation:
    """SPEC §3.8 契约 C8：单源失败不中断主流程。"""

    async def test_provider_exception_is_ignored(self) -> None:
        broken = StubProvider("crossref", error=RuntimeError("boom"))
        working = StubProvider("openalex", patch=SourcePatch(year=2024))
        resolver = MetadataResolverImpl([broken, working])

        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.year == 2024

    async def test_all_providers_failing_returns_patch(self) -> None:
        resolver = MetadataResolverImpl(
            [
                StubProvider("crossref", error=RuntimeError("boom")),
                StubProvider("openalex", error=RuntimeError("boom")),
            ]
        )
        patch = SourcePatch(title="Local Only")
        assert await resolver.enrich(patch) == patch

    async def test_slow_provider_times_out(self) -> None:
        slow = StubProvider("crossref", patch=SourcePatch(venue="Too Slow"), delay=1.0)
        fast = StubProvider("openalex", patch=SourcePatch(year=2024))
        resolver = MetadataResolverImpl([slow, fast], timeout_s=0.05)

        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.year == 2024
        assert result.venue is None, "超时的来源不得贡献字段"

    async def test_providers_are_queried_concurrently(self) -> None:
        import time

        providers = [
            StubProvider(f"p{index}", patch=SourcePatch(abstract=f"a{index}"), delay=0.15)
            for index in range(4)
        ]
        resolver = MetadataResolverImpl(providers, timeout_s=5.0)

        started = time.perf_counter()
        await resolver.enrich(SourcePatch(title="t"))
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5, f"四个来源应当并发查询，实际耗时 {elapsed:.2f}s"


class TestDeterminism:
    async def test_ties_are_broken_by_provider_name(self) -> None:
        """置信度与优先级都相同时必须给出确定结果，

        否则同一次元数据补全会在不同运行间因并发完成顺序不同而产生不同字段值。
        """
        alpha = StubProvider("alpha", patch=SourcePatch(abstract="from alpha"))
        beta = StubProvider("beta", patch=SourcePatch(abstract="from beta"))
        resolver = MetadataResolverImpl([beta, alpha])

        results = {(await resolver.enrich(SourcePatch(title="t"))).abstract for _ in range(5)}
        assert results == {"from alpha"}


class TestRetraction:
    async def test_only_true_is_adopted(self) -> None:
        """撤稿是单向的：没有撤稿库的来源不该有能力"洗白"已知撤稿。"""
        claims_clean = StubProvider("crossref", patch=SourcePatch(retracted=False))
        claims_retracted = StubProvider("retraction", patch=SourcePatch(retracted=True))
        resolver = MetadataResolverImpl([claims_clean, claims_retracted])
        assert (await resolver.enrich(SourcePatch(doi="10.1/x"))).retracted is True

    async def test_false_alone_stays_none(self) -> None:
        resolver = MetadataResolverImpl(
            [StubProvider("crossref", patch=SourcePatch(retracted=False, venue="V"))]
        )
        result = await resolver.enrich(SourcePatch(title="t"))
        assert result.retracted is None
        assert result.venue == "V"

    async def test_local_snapshot_matches_doi(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "retractions.csv"
        csv_path.write_text(
            "OriginalPaperDOI,Title\n10.5555/retracted.2020.001,A Withdrawn Study\n",
            encoding="utf-8",
        )
        provider = RetractionProvider(csv_path)
        assert provider.size == 1

        match = await provider.lookup(SourcePatch(doi="https://doi.org/10.5555/RETRACTED.2020.001"))
        assert match is not None
        assert match.patch.retracted is True
        assert match.confidence == "doi"

    async def test_local_snapshot_miss(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "retractions.csv"
        csv_path.write_text("OriginalPaperDOI\n10.5555/other\n", encoding="utf-8")
        provider = RetractionProvider(csv_path)
        assert await provider.lookup(SourcePatch(doi="10.5555/clean.2020.001")) is None

    async def test_no_doi_returns_none(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "retractions.csv"
        csv_path.write_text("OriginalPaperDOI\n10.5555/x\n", encoding="utf-8")
        provider = RetractionProvider(csv_path)
        assert await provider.lookup(SourcePatch(title="no doi here")) is None

    async def test_unconfigured_snapshot_contributes_nothing(self) -> None:
        """未配置快照时安静地不贡献——而不是"查不到就当作未撤稿"。

        后者是把未知当成已知，会让用户以为已经做过撤稿检查。
        """
        provider = RetractionProvider(None)
        assert provider.size == 0
        assert await provider.lookup(SourcePatch(doi="10.1/x")) is None

    async def test_missing_file_degrades(self, tmp_path: Path) -> None:
        provider = RetractionProvider(tmp_path / "nope.csv")
        assert provider.size == 0
        assert await provider.lookup(SourcePatch(doi="10.1/x")) is None

    async def test_malformed_csv_does_not_raise(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "bad.csv"
        csv_path.write_bytes(b"\xff\xfe\x00binary garbage")
        provider = RetractionProvider(csv_path)
        assert await provider.lookup(SourcePatch(doi="10.1/x")) is None

    async def test_alternative_column_names(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "alt.csv"
        csv_path.write_text("doi,title\n10.5555/alt.2020.001,Alt Title\n", encoding="utf-8")
        provider = RetractionProvider(csv_path)
        assert provider.size == 1


class TestClose:
    async def test_closes_every_provider(self) -> None:
        providers = [StubProvider("a"), StubProvider("b")]
        resolver = MetadataResolverImpl(providers)
        await resolver.aclose()
        assert all(provider.closed for provider in providers)

    async def test_close_failure_does_not_stop_others(self) -> None:
        class BadClose(StubProvider):
            async def aclose(self) -> None:
                raise RuntimeError("cannot close")

        good = StubProvider("good")
        resolver = MetadataResolverImpl([BadClose("bad"), good])
        await resolver.aclose()
        assert good.closed is True


class TestProviderContract:
    def test_resolver_satisfies_port(self) -> None:
        from scitrace.ports import MetadataResolver

        assert isinstance(MetadataResolverImpl([]), MetadataResolver)

    def test_retraction_satisfies_port(self) -> None:
        from scitrace.ports import MetadataProvider

        assert isinstance(RetractionProvider(None), MetadataProvider)

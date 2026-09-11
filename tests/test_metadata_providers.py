"""三个学术元数据来源适配器的测试。

**全部请求由 :class:`httpx.MockTransport` 拦截，测试不联网。**
这不只是为了快：元数据补全的失败模式（限流、5xx、非 JSON 响应、字段缺失）
恰恰是最需要被反复验证、又在真实网络里最难稳定复现的部分。用固定的
响应体把"429 之后再成功""``issued`` 为 null""``externalIds`` 为 null"
变成普通的参数化测试，是这些行为唯一可靠的回归防线。

测试数据全部为虚构：虚构 DOI（``10.4242/...``）、虚构作者、虚构期刊、
``example.org`` 域名。审计脚本会把本项目与参考实现的字符串字面量逐一比对，
使用真实论文数据（尤其是真实 DOI 与作者名）只会制造无意义的相似命中。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from scitrace.adapters.llm.retry import RetryPolicy, is_retryable_error
from scitrace.adapters.metadata.crossref import CrossrefProvider
from scitrace.adapters.metadata.http_base import (
    TITLE_MATCH_THRESHOLD,
    HttpMetadataProvider,
    HttpStatusError,
    dig,
    encode_doi_path,
    strip_markup,
    title_similarity,
)
from scitrace.adapters.metadata.openalex import OpenAlexProvider
from scitrace.adapters.metadata.semantic_scholar import SemanticScholarProvider
from scitrace.domain import SourcePatch
from scitrace.ports import MetadataMatch, MetadataProvider

# --------------------------------------------------------------------------- #
# 虚构测试数据
# --------------------------------------------------------------------------- #

DOI = "10.4242/scitrace.fixture.2024.0001"
DOI_UPPER_URL = "https://doi.org/10.4242/SCITRACE.FIXTURE.2024.0001"
TITLE = "Adaptive Evidence Retrieval for Scientific Question Answering"
#: 与 :data:`TITLE` 仅差大小写与句末标点 → 相似度 1.0。
TITLE_CASE_VARIANT = "adaptive evidence retrieval for scientific question answering."
#: 加长副标题 → 相似度 ≈ 0.857（≥ 阈值，应当命中）。
TITLE_EXTENDED = "Adaptive Evidence Retrieval for Scientific Question Answering: A Benchmark"
#: 同主题不同工作 → 相似度 0.5（< 阈值，必须放弃）。
TITLE_NEAR_MISS = "Adaptive Evidence Extraction for Clinical Question Answering"
#: 完全无关 → 相似度 0.0。
TITLE_UNRELATED = "Medieval Poetry and Courtly Love in the Low Countries"
AUTHORS = ["Dana R. Whitfield", "Priya N. Raman"]
VENUE = "Journal of Example Studies"
ABSTRACT_JATS = (
    "<jats:p>We introduce a <jats:italic>fixture-only</jats:italic> benchmark "
    "for evidence tracing.</jats:p>"
)
ABSTRACT_PLAIN = "We introduce a fixture-only benchmark for evidence tracing."

#: 重试策略：等待时间置零，并由 :class:`SleepSpy` 接管睡眠，
#: 使"429 之后成功"这类测试瞬时完成且可断言重试次数。
FAST_RETRY = RetryPolicy(attempts=3, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0)


class SleepSpy:
    """记录重试等待时长但不真的等待。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# --------------------------------------------------------------------------- #
# MockTransport 工具
# --------------------------------------------------------------------------- #


def json_responder(
    payload: Any,
    *,
    status: int = 200,
    record: list[httpx.Request] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """恒定返回同一个 JSON 响应。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(status, json=payload)

    return handler


def flaky_handler(
    payload: Any,
    *,
    fail_times: int,
    fail_status: int = 429,
    record: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    """前 ``fail_times`` 次返回 ``fail_status``，之后返回 200 + ``payload``。"""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        record.append(request)
        state["calls"] += 1
        if state["calls"] <= fail_times:
            return httpx.Response(fail_status, json={"error": "rate limited"})
        return httpx.Response(200, json=payload)

    return handler


def text_responder(
    body: str,
    *,
    status: int = 200,
    content_type: str = "text/html",
    record: list[httpx.Request] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """返回非 JSON 响应体（用于"响应不是 JSON"的降级测试）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return handler


def failing_responder(
    error: Exception, *, record: list[httpx.Request] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """抛出传输层异常（连接失败、超时等）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        raise error

    return handler


@asynccontextmanager
async def open_provider(
    provider_cls: type[HttpMetadataProvider],
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleep: SleepSpy | None = None,
    **kwargs: Any,
) -> AsyncIterator[HttpMetadataProvider]:
    """构造一个走 MockTransport 的 provider，并在退出时关闭测试客户端。"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = provider_cls(
        client=client,
        sleep=sleep if sleep is not None else SleepSpy(),
        retry_policy=FAST_RETRY,
        **kwargs,
    )
    try:
        yield provider
    finally:
        await client.aclose()


def only_request(requests: list[httpx.Request]) -> httpx.Request:
    """断言"恰好发出一次请求"并返回它。"""
    assert len(requests) == 1, f"期望 1 次请求，实际 {len(requests)} 次"
    return requests[0]


# --------------------------------------------------------------------------- #
# 各来源的虚构响应体
# --------------------------------------------------------------------------- #


def crossref_work(**overrides: Any) -> dict[str, Any]:
    """构造一个 Crossref 作品对象（``message`` 里的那一层）。"""
    work: dict[str, Any] = {
        "DOI": DOI,
        "title": [TITLE],
        "author": [
            {"given": "Dana R.", "family": "Whitfield", "sequence": "first"},
            {"given": "Priya N.", "family": "Raman", "sequence": "additional"},
        ],
        "issued": {"date-parts": [[2024, 5, 17]]},
        "container-title": [VENUE],
        "is-referenced-by-count": 37,
        "abstract": ABSTRACT_JATS,
    }
    work.update(overrides)
    return work


def crossref_doi_payload(**overrides: Any) -> dict[str, Any]:
    """``/works/{doi}`` 的完整响应体。"""
    return {"status": "ok", "message-type": "work", "message": crossref_work(**overrides)}


def crossref_search_payload(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``/works?query.bibliographic=`` 的完整响应体。"""
    if items is None:
        items = [
            crossref_work(
                DOI="10.4242/scitrace.fixture.2024.0002",
                title=[TITLE_NEAR_MISS],
                **{"is-referenced-by-count": 5},
            ),
            crossref_work(
                DOI="10.4242/scitrace.fixture.2024.0003",
                title=[TITLE_EXTENDED],
                **{"is-referenced-by-count": 11},
            ),
            crossref_work(
                DOI="10.4242/scitrace.fixture.2024.0004",
                title=[TITLE_UNRELATED],
                **{"is-referenced-by-count": 2},
            ),
        ]
    return {
        "status": "ok",
        "message-type": "work-list",
        "message": {"total-results": len(items), "items": items},
    }


def s2_paper(**overrides: Any) -> dict[str, Any]:
    """构造一个 Semantic Scholar 作品对象。"""
    paper: dict[str, Any] = {
        "paperId": "0f1e2d3c4b5a6978",
        "title": TITLE,
        "year": 2024,
        "venue": VENUE,
        "abstract": ABSTRACT_PLAIN,
        "citationCount": 37,
        "externalIds": {"DOI": DOI, "CorpusId": 424242},
        "authors": [
            {"authorId": "1001", "name": "Dana R. Whitfield"},
            {"authorId": "1002", "name": "Priya N. Raman"},
        ],
    }
    paper.update(overrides)
    return paper


def s2_search_payload(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``/paper/search`` 的完整响应体。"""
    if items is None:
        items = [
            s2_paper(
                paperId="aaaa1111bbbb2222",
                title=TITLE_NEAR_MISS,
                citationCount=5,
                externalIds={"DOI": "10.4242/scitrace.fixture.2024.0002"},
            ),
            s2_paper(
                paperId="cccc3333dddd4444",
                title=TITLE_EXTENDED,
                citationCount=11,
                externalIds={"DOI": "10.4242/scitrace.fixture.2024.0003"},
            ),
            s2_paper(
                paperId="eeee5555ffff6666",
                title=TITLE_UNRELATED,
                citationCount=2,
                externalIds={"DOI": "10.4242/scitrace.fixture.2024.0004"},
            ),
        ]
    return {"total": len(items), "offset": 0, "data": items}


def openalex_work(**overrides: Any) -> dict[str, Any]:
    """构造一个 OpenAlex 作品对象。"""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W0000000001",
        "doi": DOI_UPPER_URL,
        "title": TITLE,
        "display_name": TITLE,
        "publication_year": 2024,
        "cited_by_count": 37,
        "type": "article",
        "open_access": {
            "is_oa": True,
            "oa_status": "gold",
            "oa_url": "https://repository.example.org/items/fixture-0001.pdf",
        },
        "authorships": [
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A0000000001",
                    "display_name": "Dana R. Whitfield",
                },
            },
            {
                "author_position": "additional",
                "author": {
                    "id": "https://openalex.org/A0000000002",
                    "display_name": "Priya N. Raman",
                },
            },
        ],
        "primary_location": {
            "is_oa": True,
            "source": {
                "id": "https://openalex.org/S0000000001",
                "display_name": VENUE,
                "type": "journal",
                "is_in_doaj": True,
            },
        },
    }
    work.update(overrides)
    return work


def openalex_search_payload(items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """``/works?search=`` 的完整响应体。"""
    if items is None:
        items = [
            openalex_work(
                id="https://openalex.org/W0000000002",
                doi="https://doi.org/10.4242/scitrace.fixture.2024.0002",
                title=TITLE_NEAR_MISS,
                display_name=TITLE_NEAR_MISS,
                cited_by_count=5,
            ),
            openalex_work(
                id="https://openalex.org/W0000000003",
                doi="https://doi.org/10.4242/scitrace.fixture.2024.0003",
                title=TITLE_EXTENDED,
                display_name=TITLE_EXTENDED,
                cited_by_count=11,
            ),
            openalex_work(
                id="https://openalex.org/W0000000004",
                doi="https://doi.org/10.4242/scitrace.fixture.2024.0004",
                title=TITLE_UNRELATED,
                display_name=TITLE_UNRELATED,
                cited_by_count=2,
            ),
        ]
    return {"meta": {"count": len(items), "per_page": 5}, "results": items}


@dataclass(frozen=True)
class SourceSpec:
    """一个来源的最小可用访问方式，用于跨来源的参数化测试。"""

    provider_cls: type[HttpMetadataProvider]
    doi_payload: dict[str, Any]
    search_payload: dict[str, Any]


#: 三个来源的类，用于"公共行为"类的参数化（协议一致性、客户端归属、超时校验）。
PROVIDER_CLASSES = [CrossrefProvider, SemanticScholarProvider, OpenAlexProvider]

SOURCES = [
    pytest.param(
        SourceSpec(CrossrefProvider, crossref_doi_payload(), crossref_search_payload()),
        id="crossref",
    ),
    pytest.param(
        SourceSpec(SemanticScholarProvider, s2_paper(), s2_search_payload()),
        id="semantic_scholar",
    ),
    pytest.param(
        SourceSpec(OpenAlexProvider, openalex_work(), openalex_search_payload()),
        id="openalex",
    ),
]


# --------------------------------------------------------------------------- #
# 标题相似度
# --------------------------------------------------------------------------- #


class TestTitleSimilarity:
    """token 级 Jaccard 相似度（SPEC §3.5 的"rapidfuzz-free 自实现"）。"""

    def test_identical_titles_score_one(self) -> None:
        assert title_similarity(TITLE, TITLE) == 1.0

    def test_case_and_punctuation_differences_are_nearly_free(self) -> None:
        assert title_similarity(TITLE, TITLE_CASE_VARIANT) >= 0.99

    def test_word_order_is_ignored(self) -> None:
        """集合语义：词序调换不影响得分（这是取舍，不是缺陷）。"""
        shuffled = "Question Answering for Scientific Evidence Retrieval Adaptive"
        assert title_similarity(shuffled, TITLE) == 1.0

    def test_stopword_differences_do_not_matter(self) -> None:
        assert title_similarity("The Study of X", "Study of X") == 1.0

    def test_completely_different_titles_score_low(self) -> None:
        assert title_similarity(TITLE, TITLE_UNRELATED) <= 0.1

    def test_near_miss_stays_below_threshold(self) -> None:
        """同主题不同工作必须落在阈值之下——这是"宁可不补"的第一道防线。"""
        score = title_similarity(TITLE, TITLE_NEAR_MISS)
        assert score < TITLE_MATCH_THRESHOLD
        assert score == pytest.approx(0.5)

    def test_subtitle_extension_stays_above_threshold(self) -> None:
        score = title_similarity(TITLE, TITLE_EXTENDED)
        assert score >= TITLE_MATCH_THRESHOLD
        assert score == pytest.approx(6 / 7)

    def test_chinese_titles_use_character_bigrams(self) -> None:
        """中文没有空格可依，按字符二元组切分；措辞相近的标题应得高分。"""
        score = title_similarity(
            "面向科研文献的证据可溯源问答", "面向科研文献的证据抽取问答"
        )
        assert score > 0.5

    def test_chinese_and_latin_do_not_cross_match(self) -> None:
        assert title_similarity("面向科研文献的证据可溯源问答", TITLE) == 0.0

    def test_empty_input_scores_zero(self) -> None:
        assert title_similarity("", "") == 0.0
        assert title_similarity("", TITLE) == 0.0
        assert title_similarity(TITLE, "") == 0.0

    def test_short_and_stopword_only_titles_score_zero(self) -> None:
        """整句都是停用词/短词时有效 token 集为空，不能因为"都为空"而判为相似。"""
        assert title_similarity("The Of An", "of the an") == 0.0

    def test_threshold_matches_spec(self) -> None:
        assert TITLE_MATCH_THRESHOLD == 0.8


class TestHttpBaseHelpers:
    """基类里那些"看起来琐碎但决定成败"的小工具。"""

    def test_status_error_is_retryable_only_for_retryable_codes(self) -> None:
        """``HttpStatusError`` 存在的唯一理由：让重试策略看得见状态码。"""
        assert is_retryable_error(HttpStatusError(429, "u")) is True
        assert is_retryable_error(HttpStatusError(503, "u")) is True
        assert is_retryable_error(HttpStatusError(404, "u")) is False
        assert is_retryable_error(HttpStatusError(400, "u")) is False

    def test_encode_doi_path_escapes_delimiters_but_keeps_slashes(self) -> None:
        assert encode_doi_path("10.4242/a#b c?d") == "10.4242/a%23b%20c%3Fd"
        assert encode_doi_path(DOI) == DOI

    def test_dig_tolerates_missing_and_mistyped_paths(self) -> None:
        payload = {"a": [{"b": {"c": 7}}]}
        assert dig(payload, "a", 0, "b", "c") == 7
        assert dig(payload, "a", 9, "b") is None
        assert dig(payload, "a", 0, "b", "c", "d") is None
        assert dig(payload, "missing", 0) is None
        assert dig(None, "a") is None

    def test_strip_markup_handles_jats_and_escaped_tags(self) -> None:
        assert strip_markup(ABSTRACT_JATS) == ABSTRACT_PLAIN
        assert strip_markup("&lt;jats:p&gt;Escaped&lt;/jats:p&gt;") == "Escaped"
        assert strip_markup(None) is None
        assert strip_markup("   ") is None

    def test_strip_markup_keeps_math_comparisons(self) -> None:
        """``<`` 后面不是字母，就不是标签——剥标签不该破坏正常文本。"""
        assert strip_markup("accuracy < 0.5 and recall > 0.9") == "accuracy < 0.5 and recall > 0.9"


# --------------------------------------------------------------------------- #
# 跨来源的公共行为
# --------------------------------------------------------------------------- #


class TestProviderContract:
    """三个来源都满足 :class:`~scitrace.ports.MetadataProvider` 契约。"""

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_satisfies_protocol(self, provider_cls: type[HttpMetadataProvider]) -> None:
        provider = provider_cls()
        try:
            assert isinstance(provider, MetadataProvider)
            assert provider.name.islower()
            assert provider.name
        finally:
            await provider.aclose()

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    def test_source_name_is_lowercase_identifier(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        assert provider_cls.SOURCE_NAME == provider_cls.SOURCE_NAME.lower()

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_owns_and_closes_its_own_client(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        provider = provider_cls()
        assert isinstance(provider.client, httpx.AsyncClient)
        assert not provider.client.is_closed
        await provider.aclose()
        assert provider.client.is_closed

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_does_not_close_injected_client(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        """注入的客户端可能被多个 provider 共享，谁都不能替别人关掉它。"""
        client = httpx.AsyncClient(transport=httpx.MockTransport(json_responder({})))
        provider = provider_cls(client=client)
        await provider.aclose()
        assert not client.is_closed
        await client.aclose()

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_no_clue_means_no_request(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        """既无 DOI 也无标题时直接返回 None，不浪费一次请求。"""
        requests: list[httpx.Request] = []
        async with open_provider(provider_cls, json_responder({}, record=requests)) as provider:
            assert await provider.lookup(SourcePatch(year=2024)) is None
        assert requests == []

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_lookup_does_not_mutate_patch(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        patch = SourcePatch(doi=DOI_UPPER_URL, title=TITLE)
        before = patch.model_dump()
        async with open_provider(provider_cls, json_responder({}, status=500)) as provider:
            await provider.lookup(patch)
        assert patch.model_dump() == before

    @pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES)
    async def test_invalid_timeout_rejected_at_construction(
        self, provider_cls: type[HttpMetadataProvider]
    ) -> None:
        with pytest.raises(ValueError):
            provider_cls(timeout_s=0)


@pytest.mark.parametrize("spec", SOURCES)
class TestDegradation:
    """SPEC §3.5 的硬性要求：任何失败都只降级，不抛出。"""

    async def test_http_500_returns_none(self, spec: SourceSpec) -> None:
        requests: list[httpx.Request] = []
        handler = json_responder({"error": "server on fire"}, status=500, record=requests)
        async with open_provider(spec.provider_cls, handler) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None
        assert len(requests) >= 1

    async def test_http_429_exhausted_returns_none(self, spec: SourceSpec) -> None:
        """重试次数用尽后仍然只返回 None，不把异常抛给组合层。"""
        requests: list[httpx.Request] = []
        sleep = SleepSpy()
        handler = json_responder({"error": "slow down"}, status=429, record=requests)
        async with open_provider(spec.provider_cls, handler, sleep=sleep) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None
        assert len(requests) == FAST_RETRY.attempts
        assert len(sleep.delays) == FAST_RETRY.attempts - 1

    async def test_non_json_response_returns_none(self, spec: SourceSpec) -> None:
        """非 JSON 响应不重试：同样的响应体再取一次也不会变成合法 JSON。"""
        requests: list[httpx.Request] = []
        handler = text_responder("<html>gateway error</html>", record=requests)
        async with open_provider(spec.provider_cls, handler) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None
        assert len(requests) == 1

    async def test_connection_error_returns_none(self, spec: SourceSpec) -> None:
        request = httpx.Request("GET", "https://api.example.org/works")
        handler = failing_responder(httpx.ConnectError("connection refused", request=request))
        async with open_provider(spec.provider_cls, handler) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None

    async def test_timeout_returns_none(self, spec: SourceSpec) -> None:
        request = httpx.Request("GET", "https://api.example.org/works")
        handler = failing_responder(httpx.ReadTimeout("timed out", request=request))
        async with open_provider(spec.provider_cls, handler) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_retryable_failure_then_success(self, spec: SourceSpec) -> None:
        """429 之后返回 200：最终应当拿到结果（限流是常态，不是终点）。"""
        requests: list[httpx.Request] = []
        sleep = SleepSpy()
        handler = flaky_handler(spec.doi_payload, fail_times=2, record=requests)
        async with open_provider(spec.provider_cls, handler, sleep=sleep) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.confidence == "doi"
        assert len(requests) == 3
        assert len(sleep.delays) == 2

    async def test_404_is_not_retried(self, spec: SourceSpec) -> None:
        """404 重试没有意义：一次请求后就该放弃，而不是白等两次退避。"""
        requests: list[httpx.Request] = []
        sleep = SleepSpy()
        handler = json_responder({"error": "not found"}, status=404, record=requests)
        async with open_provider(spec.provider_cls, handler, sleep=sleep) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None
        assert len(requests) == 1
        assert sleep.delays == []

    async def test_response_is_not_an_object_returns_none(self, spec: SourceSpec) -> None:
        """响应是 JSON 但不是对象（例如裸数组）时同样降级。"""
        async with open_provider(spec.provider_cls, json_responder([1, 2, 3])) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None

    async def test_doi_lookup_end_to_end(self, spec: SourceSpec) -> None:
        async with open_provider(spec.provider_cls, json_responder(spec.doi_payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI_UPPER_URL))
        assert isinstance(match, MetadataMatch)
        assert match.provider == spec.provider_cls.SOURCE_NAME
        assert match.confidence == "doi"
        assert match.patch.title == TITLE
        assert match.patch.authors == AUTHORS
        assert match.patch.year == 2024
        assert match.patch.citation_count == 37

    async def test_title_search_end_to_end(self, spec: SourceSpec) -> None:
        """标题检索命中相似度最高的候选（0.857），而不是第一个候选（0.5）。"""
        handler = json_responder(spec.search_payload)
        async with open_provider(spec.provider_cls, handler) as provider:
            match = await provider.lookup(SourcePatch(title=TITLE))
        assert match is not None
        assert match.confidence == "title"
        assert match.score == pytest.approx(6 / 7)
        assert match.matched_title == TITLE_EXTENDED
        assert match.patch.doi == "10.4242/scitrace.fixture.2024.0003"

    async def test_title_search_below_threshold_returns_none(self, spec: SourceSpec) -> None:
        """只有低分候选（0.5 / 0.0）时放弃补全（宁可不补，也不补错）。"""
        candidates = _candidates_of(spec)
        low_scores = _payload_with_items(spec, [candidates[0], candidates[2]])
        async with open_provider(spec.provider_cls, json_responder(low_scores)) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_title_search_empty_result_returns_none(self, spec: SourceSpec) -> None:
        async with open_provider(
            spec.provider_cls, json_responder(_payload_with_items(spec, []))
        ) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_title_search_missing_container_returns_none(self, spec: SourceSpec) -> None:
        """检索端点的候选容器字段整个缺失（``items`` / ``data`` / ``results``）。"""
        handler = json_responder({"unexpected": True})
        async with open_provider(spec.provider_cls, handler) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_blank_title_is_not_searched(self, spec: SourceSpec) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            spec.provider_cls, json_responder({"unexpected": True}, record=requests)
        ) as provider:
            assert await provider.lookup(SourcePatch(title="   ")) is None
        assert requests == []


def _candidates_of(spec: SourceSpec) -> list[dict[str, Any]]:
    """从各来源的检索响应体里取回候选列表（用于按需裁剪候选）。"""
    provider_cls = spec.provider_cls
    if provider_cls is CrossrefProvider:
        return list(spec.search_payload["message"]["items"])
    if provider_cls is SemanticScholarProvider:
        return list(spec.search_payload["data"])
    return list(spec.search_payload["results"])


def _payload_with_items(spec: SourceSpec, items: list[dict[str, Any]]) -> dict[str, Any]:
    """用给定候选重建检索响应体。"""
    provider_cls = spec.provider_cls
    if provider_cls is CrossrefProvider:
        return crossref_search_payload(items)
    if provider_cls is SemanticScholarProvider:
        return s2_search_payload(items)
    return openalex_search_payload(items)


class TestUnexpectedSubclassErrors:
    """子类实现里的缺陷也必须在公开入口被降级。

    SPEC §3.5 的降级要求针对的是"任一 provider 失败不影响主流程"，
    因此它保护的不只是网络错误——一个写错的字段映射同样不能把整次摄入拖垮。
    这些测试钉住的正是 :meth:`HttpMetadataProvider.lookup` 的"绝不抛出"契约。
    """

    async def test_mapping_bug_is_swallowed(self) -> None:
        class ExplodingMapping(CrossrefProvider):
            def _patch_from_payload(self, payload: Mapping[str, Any]) -> SourcePatch:
                raise RuntimeError("字段映射写错了")

        handler = json_responder(crossref_doi_payload())
        async with open_provider(ExplodingMapping, handler) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None

    async def test_transport_bug_in_subclass_is_swallowed(self) -> None:
        class ExplodingFetch(CrossrefProvider):
            async def _fetch_by_doi(self, doi: str) -> Mapping[str, Any] | None:
                raise RuntimeError("子类忘了走 _request_json")

        handler = json_responder(crossref_doi_payload())
        async with open_provider(ExplodingFetch, handler) as provider:
            assert await provider.lookup(SourcePatch(doi=DOI)) is None

    def test_abstract_methods_cannot_be_left_unimplemented(self) -> None:
        """ABC 让"忘了实现子类接口"在构造期就失败，而不是伪装成空结果。"""

        class Incomplete(HttpMetadataProvider):
            SOURCE_NAME = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# Crossref
# --------------------------------------------------------------------------- #


class TestCrossrefProvider:
    """Crossref 的字段映射与端点选择。"""

    async def test_doi_lookup_maps_every_field(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            CrossrefProvider, json_responder(crossref_doi_payload(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.provider == "crossref"
        assert match.confidence == "doi"
        assert match.matched_title == TITLE
        assert match.patch.title == TITLE
        assert match.patch.authors == AUTHORS
        assert match.patch.year == 2024
        assert match.patch.venue == VENUE
        assert match.patch.doi == DOI
        assert match.patch.citation_count == 37

        request = only_request(requests)
        assert str(request.url).startswith("https://api.crossref.org/works/")
        assert DOI in str(request.url)

    async def test_jats_markup_is_stripped_from_abstract(self) -> None:
        """``<jats:p>`` 这类标签绝不能进入 Source.abstract。"""
        handler = json_responder(crossref_doi_payload())
        async with open_provider(CrossrefProvider, handler) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.abstract == ABSTRACT_PLAIN
        assert "<jats:" not in (match.patch.abstract or "")

    async def test_title_search_reports_score_and_confidence(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            CrossrefProvider, json_responder(crossref_search_payload(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(title=TITLE, year=2024))

        assert match is not None
        assert match.confidence == "title"
        assert match.score == pytest.approx(6 / 7)
        assert match.matched_title == TITLE_EXTENDED
        assert match.patch.doi == "10.4242/scitrace.fixture.2024.0003"

        request = only_request(requests)
        assert request.url.params.get("query.bibliographic") == TITLE
        assert request.url.params.get("rows") == "5"

    async def test_title_search_without_close_candidate_returns_none(self) -> None:
        payload = crossref_search_payload(
            [crossref_work(title=[TITLE_NEAR_MISS]), crossref_work(title=[TITLE_UNRELATED])]
        )
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_doi_takes_precedence_over_title(self) -> None:
        """有 DOI 时只走精确端点，不做标题检索（结果必须可复现）。"""
        requests: list[httpx.Request] = []
        async with open_provider(
            CrossrefProvider, json_responder(crossref_doi_payload(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI, title=TITLE_NEAR_MISS))

        assert match is not None
        assert match.confidence == "doi"
        assert match.patch.title == TITLE
        request = only_request(requests)
        assert "query.bibliographic" not in str(request.url)

    async def test_missing_nested_fields_are_tolerated(self) -> None:
        """``issued`` / ``container-title`` / ``author`` 缺失时不得抛异常。"""
        payload = crossref_doi_payload(
            issued=None,
            author=None,
            **{"container-title": []},
        )
        payload["message"].pop("abstract")
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.title == TITLE
        assert match.patch.doi == DOI
        assert match.patch.year is None
        assert match.patch.venue is None
        assert match.patch.authors is None
        assert match.patch.abstract is None

    async def test_structurally_broken_fields_are_tolerated(self) -> None:
        """字段类型整个不对（列表变字符串、年份是占位值）也不得抛异常。"""
        payload = crossref_doi_payload(
            title=12345,
            author={"given": "not-a-list"},
            issued={"date-parts": [[0]]},
            **{"container-title": [None, ""], "is-referenced-by-count": True},
        )
        payload["message"]["abstract"] = None
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.title is None
        assert match.patch.year is None
        assert match.patch.authors is None
        assert match.patch.venue is None
        assert match.patch.citation_count is None
        # 只有 DOI 可用时仍返回匹配：它至少确认了标识符本身。
        assert match.patch.provided_fields() == {"doi"}

    async def test_author_without_given_name_uses_family_only(self) -> None:
        payload = crossref_doi_payload(
            author=[
                {"family": "Whitfield"},
                {"name": "Example Standards Consortium"},
                {"given": "Priya N.", "family": "Raman"},
            ]
        )
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.authors == [
            "Whitfield",
            "Example Standards Consortium",
            "Priya N. Raman",
        ]

    async def test_issued_with_only_year(self) -> None:
        """``date-parts`` 只给年份（没有月日）是最常见的形状。"""
        payload = crossref_doi_payload(issued={"date-parts": [[2019]]})
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.year == 2019

    async def test_missing_date_parts_returns_none_year(self) -> None:
        payload = crossref_doi_payload(issued={})
        async with open_provider(CrossrefProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.year is None

    async def test_mailto_is_sent_on_both_endpoints(self) -> None:
        """``mailto`` 让 Crossref 把请求归入 polite pool（限流额度更高）。"""
        requests: list[httpx.Request] = []
        async with open_provider(
            CrossrefProvider,
            json_responder(crossref_doi_payload(), record=requests),
            mailto="ops@example.org",
        ) as provider:
            await provider.lookup(SourcePatch(doi=DOI))
            await provider.lookup(SourcePatch(title=TITLE))

        assert len(requests) == 2
        assert all(request.url.params.get("mailto") == "ops@example.org" for request in requests)

    async def test_mailto_absent_when_not_configured(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            CrossrefProvider, json_responder(crossref_doi_payload(), record=requests)
        ) as provider:
            await provider.lookup(SourcePatch(doi=DOI))
        assert "mailto=" not in str(only_request(requests).url)


# --------------------------------------------------------------------------- #
# Semantic Scholar
# --------------------------------------------------------------------------- #


class TestSemanticScholarProvider:
    """Semantic Scholar 的字段映射、Graph API 参数与限流处理。"""

    async def test_doi_lookup_maps_every_field(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            SemanticScholarProvider, json_responder(s2_paper(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.provider == "semantic_scholar"
        assert match.confidence == "doi"
        assert match.matched_title == TITLE
        assert match.patch.title == TITLE
        assert match.patch.authors == AUTHORS
        assert match.patch.year == 2024
        assert match.patch.venue == VENUE
        assert match.patch.doi == DOI
        assert match.patch.citation_count == 37
        assert match.patch.abstract == ABSTRACT_PLAIN

        request = only_request(requests)
        assert str(request.url).startswith("https://api.semanticscholar.org/graph/v1/paper/DOI:")
        assert request.url.params.get("fields") == SemanticScholarProvider.FIELDS

    async def test_title_search_reports_score_and_confidence(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            SemanticScholarProvider, json_responder(s2_search_payload(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(title=TITLE))

        assert match is not None
        assert match.confidence == "title"
        assert match.score == pytest.approx(6 / 7)
        assert match.matched_title == TITLE_EXTENDED
        assert match.patch.doi == "10.4242/scitrace.fixture.2024.0003"

        request = only_request(requests)
        assert "/paper/search" in str(request.url)
        assert request.url.params.get("query") == TITLE
        assert request.url.params.get("limit") == "5"
        assert request.url.params.get("fields") == SemanticScholarProvider.FIELDS

    async def test_title_search_without_close_candidate_returns_none(self) -> None:
        payload = s2_search_payload(
            [s2_paper(title=TITLE_NEAR_MISS), s2_paper(title=TITLE_UNRELATED)]
        )
        async with open_provider(SemanticScholarProvider, json_responder(payload)) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_external_ids_null_is_tolerated(self) -> None:
        """``externalIds`` 为 ``null``（无 DOI 的会议论文）时其余字段照常返回。"""
        payload = s2_paper(externalIds=None)
        async with open_provider(SemanticScholarProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.doi is None
        assert match.patch.title == TITLE
        assert match.patch.citation_count == 37

    async def test_missing_fields_are_tolerated(self) -> None:
        payload = s2_paper(authors=None, year=None, venue=None, abstract=None, citationCount="many")
        payload.pop("externalIds")
        payload.pop("abstract")
        async with open_provider(SemanticScholarProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.title == TITLE
        assert match.patch.authors is None
        assert match.patch.year is None
        assert match.patch.venue is None
        assert match.patch.doi is None
        assert match.patch.citation_count is None
        assert match.patch.abstract is None

    async def test_authors_without_names_are_skipped(self) -> None:
        payload = s2_paper(authors=[{"authorId": "1"}, {"name": "Priya N. Raman"}, {"name": ""}])
        async with open_provider(SemanticScholarProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.authors == ["Priya N. Raman"]

    async def test_api_key_header_sent_when_configured(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            SemanticScholarProvider,
            json_responder(s2_paper(), record=requests),
            api_key="fixture-api-key",
        ) as provider:
            await provider.lookup(SourcePatch(doi=DOI))
        assert only_request(requests).headers.get("x-api-key") == "fixture-api-key"

    async def test_api_key_header_absent_without_configuration(self) -> None:
        """没有 key 时不发空 ``x-api-key``：空值会被判为鉴权失败且不重试。"""
        requests: list[httpx.Request] = []
        async with open_provider(
            SemanticScholarProvider, json_responder(s2_paper(), record=requests)
        ) as provider:
            await provider.lookup(SourcePatch(doi=DOI))
        assert "x-api-key" not in only_request(requests).headers

    async def test_rate_limit_then_success(self) -> None:
        """429 是无 key 时的常态，必须靠重试跨过去。"""
        requests: list[httpx.Request] = []
        sleep = SleepSpy()
        handler = flaky_handler(s2_paper(), fail_times=1, fail_status=429, record=requests)
        async with open_provider(SemanticScholarProvider, handler, sleep=sleep) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.citation_count == 37
        assert len(requests) == 2
        assert len(sleep.delays) == 1


# --------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------- #


class TestOpenAlexProvider:
    """OpenAlex 的字段映射（含深层嵌套容错）与 quality_tier 规则。"""

    async def test_doi_lookup_maps_every_field(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            OpenAlexProvider, json_responder(openalex_work(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.provider == "openalex"
        assert match.confidence == "doi"
        assert match.matched_title == TITLE
        assert match.patch.title == TITLE
        assert match.patch.authors == AUTHORS
        assert match.patch.year == 2024
        assert match.patch.venue == VENUE
        assert match.patch.citation_count == 37
        assert match.patch.is_oa is True
        assert match.patch.oa_url == "https://repository.example.org/items/fixture-0001.pdf"
        assert match.patch.quality_tier == 3

        request = only_request(requests)
        assert str(request.url).startswith("https://api.openalex.org/works/doi:")
        assert "https://doi.org/" not in str(request.url)

    async def test_doi_is_normalized(self) -> None:
        """OpenAlex 返回 URL 形式且大小写不定，必须归一化后再入库。"""
        payload = openalex_work(doi=DOI_UPPER_URL)
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.doi == DOI

    async def test_doi_null_is_tolerated(self) -> None:
        payload = openalex_work(doi=None)
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.doi is None
        assert match.patch.title == TITLE

    async def test_display_name_is_used_when_title_missing(self) -> None:
        """老版本 API 只有 ``display_name``，缺失 ``title`` 时不能变成"无标题"。"""
        payload = openalex_work()
        payload.pop("title")
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.title == TITLE

    async def test_display_name_used_for_title_search(self) -> None:
        """标题检索路径同样要认 ``display_name``，否则老记录永远匹配不上。"""
        candidate = openalex_work(
            doi="https://doi.org/10.4242/scitrace.fixture.2024.0003",
            title=TITLE_EXTENDED,
            display_name=TITLE_EXTENDED,
        )
        candidate.pop("title")
        async with open_provider(
            OpenAlexProvider, json_responder(openalex_search_payload([candidate]))
        ) as provider:
            match = await provider.lookup(SourcePatch(title=TITLE))
        assert match is not None
        assert match.confidence == "title"
        assert match.matched_title == TITLE_EXTENDED

    async def test_title_search_reports_score_and_confidence(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            OpenAlexProvider, json_responder(openalex_search_payload(), record=requests)
        ) as provider:
            match = await provider.lookup(SourcePatch(title=TITLE))

        assert match is not None
        assert match.confidence == "title"
        assert match.score == pytest.approx(6 / 7)
        assert match.patch.doi == "10.4242/scitrace.fixture.2024.0003"

        request = only_request(requests)
        assert request.url.params.get("search") == TITLE
        assert request.url.params.get("per-page") == "5"

    async def test_title_search_without_close_candidate_returns_none(self) -> None:
        payload = openalex_search_payload(
            [openalex_work(title=TITLE_NEAR_MISS), openalex_work(title=TITLE_UNRELATED)]
        )
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            assert await provider.lookup(SourcePatch(title=TITLE)) is None

    async def test_missing_nested_fields_are_tolerated(self) -> None:
        """``primary_location`` / ``authorships`` / ``open_access`` 全部缺失。"""
        payload = openalex_work(primary_location=None, authorships=None, open_access=None)
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))

        assert match is not None
        assert match.patch.title == TITLE
        assert match.patch.venue is None
        assert match.patch.quality_tier is None
        assert match.patch.authors is None
        assert match.patch.is_oa is None
        assert match.patch.oa_url is None

    async def test_source_null_is_tolerated(self) -> None:
        """``primary_location`` 存在但 ``source`` 为 ``null``（仓库版本常见）。"""
        payload = openalex_work(primary_location={"is_oa": False, "source": None})
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.venue is None
        assert match.patch.quality_tier is None

    async def test_authorship_without_author_is_skipped(self) -> None:
        payload = openalex_work(
            authorships=[
                {"author_position": "first"},
                {"author": {"display_name": "Priya N. Raman"}},
                {"author": {}},
            ]
        )
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.authors == ["Priya N. Raman"]

    @pytest.mark.parametrize(
        ("primary_location", "expected"),
        [
            ({"source": {"type": "repository", "is_in_doaj": False}}, 0),
            ({"source": {"type": "repository", "is_in_doaj": True}}, 0),
            ({"source": {"type": "conference"}}, 1),
            ({"source": {"type": "proceedings"}}, 1),
            ({"source": {"type": "journal", "is_in_doaj": True}}, 3),
            ({"source": {"type": "journal", "is_in_doaj": False}}, 2),
            ({"source": {"type": "journal", "is_in_doaj": None}}, 2),
            ({"source": {"type": "journal"}}, 2),
            # type 缺失 → 最低档；DOAJ 标记不能把未知类型抬成 3。
            ({"source": {"is_in_doaj": True}}, 0),
            ({"source": {"type": None, "is_in_doaj": True}}, 0),
            ({"source": {}}, 0),
            # 无法判断 → None（不猜）。
            ({"source": {"type": "book"}}, None),
            ({"source": {"type": "ebook platform"}}, None),
            ({"source": None}, None),
            (None, None),
            # 连场所对象都没有（``{}``）同样属于"不知道"，不是"低档"。
            ({}, None),
        ],
    )
    async def test_quality_tier_branches(
        self, primary_location: dict[str, Any] | None, expected: int | None
    ) -> None:
        payload = openalex_work(primary_location=primary_location)
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.quality_tier == expected

    async def test_quality_tier_is_none_when_location_missing_entirely(self) -> None:
        payload = openalex_work()
        payload.pop("primary_location")
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.quality_tier is None

    async def test_is_oa_false_is_a_meaningful_value(self) -> None:
        """``is_oa=False`` 是"确认不开放"，不能被当成缺失而丢掉。"""
        payload = openalex_work(open_access={"is_oa": False, "oa_url": None})
        async with open_provider(OpenAlexProvider, json_responder(payload)) as provider:
            match = await provider.lookup(SourcePatch(doi=DOI))
        assert match is not None
        assert match.patch.is_oa is False
        assert match.patch.oa_url is None

    async def test_mailto_is_sent_when_configured(self) -> None:
        requests: list[httpx.Request] = []
        async with open_provider(
            OpenAlexProvider,
            json_responder(openalex_work(), record=requests),
            mailto="ops@example.org",
        ) as provider:
            await provider.lookup(SourcePatch(doi=DOI))
        assert only_request(requests).url.params.get("mailto") == "ops@example.org"

"""测试证据筛选与容错链。

筛选是"每个片段一次"的高频调用，因此也是畸形输出最集中的地方。
本文件的重点是**失败路径**：一个坏片段绝不能毁掉整批，
而"整批失败"与"丢弃一条"之间的差别，正是这类系统能不能用的分界线。
"""

from __future__ import annotations

import pytest
from fakes import FakeLLMClient
from scitrace.config import ScreeningSettings
from scitrace.domain import Evidence, Fragment, Source
from scitrace.pipeline.screening import CrossEncoderScreener, LLMScreener, coerce_relevance
from scitrace.prompts import NOT_APPLICABLE, get_prompt_set
from scitrace.ports import ScoredFragment


def fragment(index: int, text: str | None = None, source_key: str = "src-1") -> Fragment:
    body = text or f"第{index}段关于证据抽取的中文内容，长度足够用于筛选。"
    return Fragment(
        fragment_id=f"frag{index:03d}",
        source_key=source_key,
        text=body,
        chunk_index=index,
        section_path=["2 方法"],
        page_start=index + 1,
        page_end=index + 1,
        char_count=len(body),
    )


def source(key: str = "src-1", *, title: str = "证据可溯源问答方法研究") -> Source:
    return Source(
        key=key,
        content_hash="h" * 16,
        rel_path=f"corpus/{key}.pdf",
        title=title,
        authors=["张三"],
        year=2024,
        doi="10.5555/example.2024.001",
    )


def payload(summary: str, score: int) -> str:
    import json

    return json.dumps({"summary": summary, "relevance_score": score}, ensure_ascii=False)


class TestCoerceRelevance:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (8, 8),
            (8.0, 8),
            (8.4, 8),
            (8.6, 9),
            ("8", 8),
            ("8.0", 8),
            ("8/10", 8),
            ("3/5", 6),
            ("8 分", 8),
            ("评分：8", 8),
            ("relevance_score: 7", 7),
        ],
    )
    def test_parses_real_world_forms(self, value: object, expected: int) -> None:
        assert coerce_relevance(value) == expected

    @pytest.mark.parametrize("value", [None, "", "abc", "无", [], {}, True, False])
    def test_unparseable_returns_zero(self, value: object) -> None:
        assert coerce_relevance(value) == 0

    def test_out_of_range_is_clamped_not_dropped(self) -> None:
        """12 分说明模型认为它非常相关，丢掉它反而是错的。"""
        assert coerce_relevance(12) == 10
        assert coerce_relevance(-5) == 0

    def test_division_by_zero_is_safe(self) -> None:
        assert coerce_relevance("8/0") == 0


def make_screener(llm: FakeLLMClient, **overrides) -> LLMScreener:
    settings = ScreeningSettings(**overrides)
    return LLMScreener(
        llm=llm,
        prompts=get_prompt_set("zh"),
        settings=settings,
        max_evidence=10,
    )


class TestLLMScreener:
    async def test_empty_input(self) -> None:
        screener = make_screener(FakeLLMClient())
        assert await screener.screen("问题", [], sources={}) == []

    async def test_filters_by_threshold(self) -> None:
        llm = FakeLLMClient([payload("很相关", 9), payload("不太相关", 3)])
        screener = make_screener(llm, min_relevance=5)
        result = await screener.screen(
            "问题", [fragment(0), fragment(1)], sources={"src-1": source()}
        )
        assert len(result) == 1
        assert result[0].relevance == 9

    async def test_sorted_by_relevance_descending(self) -> None:
        llm = FakeLLMClient([payload("低", 6), payload("高", 10), payload("中", 8)])
        screener = make_screener(llm)
        result = await screener.screen(
            "问题", [fragment(i) for i in range(3)], sources={"src-1": source()}
        )
        assert [item.relevance for item in result] == [10, 8, 6]

    async def test_capped_at_max_evidence(self) -> None:
        llm = FakeLLMClient([payload(f"摘要{i}", 9) for i in range(6)])
        screener = LLMScreener(
            llm=llm, prompts=get_prompt_set("zh"), settings=ScreeningSettings(), max_evidence=3
        )
        result = await screener.screen(
            "问题", [fragment(i) for i in range(6)], sources={"src-1": source()}
        )
        assert len(result) == 3

    async def test_evidence_carries_citation_and_position(self) -> None:
        llm = FakeLLMClient([payload("摘要正文", 8)])
        screener = make_screener(llm)
        result = await screener.screen("问题", [fragment(2)], sources={"src-1": source()})

        item = result[0]
        assert isinstance(item, Evidence)
        assert item.citation == "(张三2024证据可溯 page 3)"
        assert item.page_label == "page 3"
        assert item.section_path == ["2 方法"]
        assert item.key.startswith("ev-")

    async def test_missing_source_renders_unknown(self) -> None:
        """元数据缺失时显示 unknown，而不是产生一个看起来像真的空引用。"""
        llm = FakeLLMClient([payload("摘要", 8)])
        screener = make_screener(llm)
        result = await screener.screen("问题", [fragment(0)], sources={})
        assert "unknown" in result[0].citation

    async def test_not_applicable_is_dropped(self) -> None:
        llm = FakeLLMClient([payload(NOT_APPLICABLE, 1)])
        screener = make_screener(llm)
        assert await screener.screen("问题", [fragment(0)], sources={"src-1": source()}) == []

    async def test_evidence_key_is_deterministic(self) -> None:
        llm = FakeLLMClient([payload("摘要", 8), payload("摘要", 8)])
        screener = make_screener(llm)
        first = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        second = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert first[0].key == second[0].key


class TestFaultTolerance:
    """SPEC §3.7 容错链与 §3.10 边界行为。"""

    async def test_unparseable_output_drops_only_that_fragment(self) -> None:
        llm = FakeLLMClient(["这不是 JSON", payload("正常的摘要", 9)])
        screener = make_screener(llm)
        result = await screener.screen(
            "问题", [fragment(0), fragment(1)], sources={"src-1": source()}
        )
        assert len(result) == 1
        assert result[0].relevance == 9
        assert screener.last_usage.parse_failures == 1

    async def test_llm_failure_drops_only_that_fragment(self) -> None:
        llm = FakeLLMClient(error=RuntimeError("rate limit"))
        screener = make_screener(llm)
        result = await screener.screen(
            "问题", [fragment(0), fragment(1)], sources={"src-1": source()}
        )
        assert result == []

    async def test_all_fragments_failing_still_returns(self) -> None:
        llm = FakeLLMClient(error=RuntimeError("boom"))
        screener = make_screener(llm)
        assert await screener.screen("问题", [fragment(0)], sources={}) == []

    async def test_usage_is_accumulated(self) -> None:
        llm = FakeLLMClient([payload("摘要", 9), payload("摘要", 8)])
        screener = make_screener(llm)
        await screener.screen("问题", [fragment(0), fragment(1)], sources={"src-1": source()})
        assert screener.last_usage.llm_calls == 2
        assert screener.last_usage.prompt_tokens == 20
        assert screener.last_usage.estimated_cost_usd == pytest.approx(0.002)

    async def test_usage_resets_between_calls(self) -> None:
        llm = FakeLLMClient([payload("摘要", 9), payload("摘要", 9)])
        screener = make_screener(llm)
        await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        await screener.screen("问题", [fragment(1)], sources={"src-1": source()})
        assert screener.last_usage.llm_calls == 1

    async def test_missing_score_field_still_keeps_evidence(self) -> None:
        """键名漂移不该丢掉一条本来可用的证据。"""
        llm = FakeLLMClient(['{"summary": "只有摘要没有评分"}'])
        screener = make_screener(llm, min_relevance=0)
        result = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert len(result) == 1

    async def test_alternative_score_key_names(self) -> None:
        llm = FakeLLMClient(['{"summary": "摘要", "relevance": 9}'])
        screener = make_screener(llm)
        result = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert result[0].relevance == 9

    async def test_fenced_json_is_parsed(self) -> None:
        llm = FakeLLMClient(['```json\n{"summary": "摘要", "relevance_score": 9}\n```'])
        screener = make_screener(llm)
        result = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert result and result[0].relevance == 9

    async def test_reasoning_tags_are_stripped(self) -> None:
        llm = FakeLLMClient(['<think>想一想</think>{"summary": "摘要", "relevance_score": 9}'])
        screener = make_screener(llm)
        result = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert result and result[0].relevance == 9


class TestPromptsUsed:
    async def test_screening_prompt_carries_question_citation_and_text(self) -> None:
        llm = FakeLLMClient([payload("摘要", 9)])
        screener = make_screener(llm)
        await screener.screen("跨模态对齐效果如何", [fragment(0)], sources={"src-1": source()})

        messages = llm.calls[0]
        combined = messages[0].content + messages[1].content
        assert "跨模态对齐效果如何" in combined
        assert "张三2024证据可溯" in combined
        assert "证据抽取" in combined

    async def test_json_mode_is_requested(self) -> None:
        llm = FakeLLMClient([payload("摘要", 9)])
        screener = make_screener(llm)
        await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert llm.kwargs[0]["json_object"] is True

    def test_name_matches_config_backend(self) -> None:
        assert make_screener(FakeLLMClient()).name == "llm"


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    async def rerank(self, query: str, fragments, *, top_n: int):
        self.calls += 1
        return [
            ScoredFragment(fragment=item, score=1.0, origin="rerank", rank=index + 1)
            for index, item in enumerate(fragments[:top_n])
        ]

    async def aclose(self) -> None:
        return None


class TestCrossEncoderScreener:
    def make(self, llm: FakeLLMClient, reranker: FakeReranker, **overrides):
        return CrossEncoderScreener(
            llm=llm,
            reranker=reranker,
            prompts=get_prompt_set("zh"),
            settings=ScreeningSettings(**overrides),
            max_evidence=10,
        )

    async def test_uses_reranker_then_summarizes(self) -> None:
        reranker = FakeReranker()
        llm = FakeLLMClient([payload(f"摘要{i}", 5) for i in range(4)])
        screener = self.make(llm, reranker)
        result = await screener.screen(
            "问题", [fragment(i) for i in range(4)], sources={"src-1": source()}
        )
        assert reranker.calls == 1
        assert len(result) == 4
        # 名次 → 评分：第 1 名得满分
        assert result[0].relevance == 10

    async def test_presummary_n_limits_llm_calls(self) -> None:
        """交叉编码器路线的成本优势正来自这里：只对收敛后的候选调用模型。"""
        reranker = FakeReranker()
        llm = FakeLLMClient(default=payload("摘要", 5))
        screener = self.make(llm, reranker, presummary_n=2)
        await screener.screen(
            "问题", [fragment(i) for i in range(6)], sources={"src-1": source()}
        )
        assert len(llm.calls) == 2

    async def test_name_matches_config_backend(self) -> None:
        assert self.make(FakeLLMClient(), FakeReranker()).name == "cross_encoder"

    async def test_unparseable_output_falls_back_to_rank_score(self) -> None:
        """这一路的相关性判断来自重排器，因此摘要解析失败仍可给分。"""
        llm = FakeLLMClient(["不是 JSON 的一段话"])
        screener = self.make(llm, FakeReranker(), min_relevance=1)
        result = await screener.screen("问题", [fragment(0)], sources={"src-1": source()})
        assert len(result) == 1
        assert screener.last_usage.parse_failures == 1

    async def test_not_applicable_is_dropped(self) -> None:
        llm = FakeLLMClient([payload(NOT_APPLICABLE, 1)])
        screener = self.make(llm, FakeReranker())
        assert await screener.screen("问题", [fragment(0)], sources={}) == []

    async def test_empty_input(self) -> None:
        assert await self.make(FakeLLMClient(), FakeReranker()).screen(
            "问题", [], sources={}
        ) == []


class TestPortCompliance:
    def test_llm_screener_satisfies_port(self) -> None:
        from scitrace.ports import EvidenceScreener

        assert isinstance(make_screener(FakeLLMClient()), EvidenceScreener)

    def test_cross_encoder_screener_satisfies_port(self) -> None:
        from scitrace.ports import EvidenceScreener

        screener = CrossEncoderScreener(
            llm=FakeLLMClient(),
            reranker=FakeReranker(),
            prompts=get_prompt_set("zh"),
            settings=ScreeningSettings(),
            max_evidence=5,
        )
        assert isinstance(screener, EvidenceScreener)

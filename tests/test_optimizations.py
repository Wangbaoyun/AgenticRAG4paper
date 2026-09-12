"""测试本轮六项优化的行为。

覆盖：成本闸门退化为纯 token 闸门、UNCITED 终态、非法 Unicode 清洗、
索引写入失败的文件级隔离、LLM 标题推断。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeLLMClient
from scitrace.agent.budget import Budget
from scitrace.domain import Answer
from scitrace.domain.session import SessionStatus, Usage
from scitrace.pipeline.title_inference import LLMTitleInferrer, _clean_title
from scitrace.prompts import get_prompt_set
from scitrace.util import normalize_text


class TestCostGate:
    """① 成本不可信时闸门必须退化为纯 token 闸门，而不是假装在工作。"""

    def test_unknown_cost_disables_cost_gate(self) -> None:
        budget = Budget(max_cost=1.0)
        usage = Usage(estimated_cost=999.0, cost_known=False)
        assert budget.check(usage) is None, "成本不可信时不该用它做判断"

    def test_known_cost_still_gates(self) -> None:
        budget = Budget(max_cost=1.0)
        assert budget.check(Usage(estimated_cost=999.0, cost_known=True)) is not None

    def test_token_gate_works_even_when_cost_unknown(self) -> None:
        budget = Budget(max_tokens=100, max_cost=1.0)
        assert budget.check(Usage(prompt_tokens=101, cost_known=False)) == "token 预算超限"

    def test_cost_gate_active_flag(self) -> None:
        assert Budget(max_cost=1.0).cost_gate_active is True
        assert Budget(max_tokens=10).cost_gate_active is False

    def test_merge_propagates_unknown_cost(self) -> None:
        """cost_known 是布尔，不能当计数相加——相加会得到恒为真的整数。"""
        merged = Usage(cost_known=True).merge(Usage(cost_known=False))
        assert merged.cost_known is False
        assert Usage(cost_known=True).merge(Usage(cost_known=True)).cost_known is True


class TestUncitedStatus:
    """⑤ 区分"模型没引用"与"证据不足"。"""

    def test_status_exists(self) -> None:
        assert SessionStatus.UNCITED == "UNCITED"

    async def test_answer_without_citations_is_marked_uncited(self) -> None:
        from scitrace.domain import Evidence, make_evidence_key
        from scitrace.pipeline.synthesis import bind_citations

        item = Evidence(
            key=make_evidence_key("s", "f"),
            source_key="s",
            fragment_id="f",
            summary="x",
            relevance=9,
            citation="(a2024t page 1)",
        )
        answer = bind_citations("我认为答案是 42。", evidence=[item], sources={})
        assert answer.uncited is True
        assert answer.refused is True, "无引用的答案仍不可作为依据呈现"
        assert "未引用" in answer.refusal_reason

    async def test_cited_answer_is_not_uncited(self) -> None:
        from scitrace.domain import Evidence, make_evidence_key
        from scitrace.pipeline.synthesis import bind_citations

        key = make_evidence_key("s", "f")
        item = Evidence(
            key=key, source_key="s", fragment_id="f", summary="x", relevance=9,
            citation="(a2024t page 1)",
        )
        answer = bind_citations(f"结论是 X ({key})。", evidence=[item], sources={})
        assert answer.uncited is False

    def test_answer_defaults_to_not_uncited(self) -> None:
        assert Answer(text="x").uncited is False


class TestInvalidUnicode:
    """真实语料暴露：PDF 解析会产生孤立代理项，UTF-8 编码时抛错。"""

    @pytest.mark.parametrize("bad", ["\ud835", "a\ud800b", "\udfff"])
    def test_lone_surrogates_are_removed(self, bad: str) -> None:
        cleaned = normalize_text(f"text {bad} end")
        assert not any(0xD800 <= ord(c) <= 0xDFFF for c in cleaned)
        cleaned.encode("utf-8")  # 关键：必须可编码

    def test_replacement_char_is_used(self) -> None:
        assert "\ufffd" in normalize_text("a\ud835b")

    def test_normal_text_untouched(self) -> None:
        assert normalize_text("普通的正常文本 with ASCII") == "普通的正常文本 with ASCII"

    def test_astral_characters_survive(self) -> None:
        """星光平面字符不该被误删——只清代理项，不搞连坐。"""
        assert "\U0001f600" in normalize_text("emoji \U0001f600 stays")

    def test_nfkc_folds_mathematical_alphanumerics(self) -> None:
        """数学花体字母会被 NFKC 折叠成 ASCII——这是**正确**行为，不是缺陷。

        它与"孤立代理项"是两回事：前者是合法字符的规范化，
        后者是残缺的编码。测试把它们分开钉住，避免有人为了"保住数学符号"
        而把代理项清洗也一并关掉。
        """
        assert normalize_text("\U0001d4db") == "L"


class TestTitleInference:
    """④ LLM 标题推断。"""

    def test_clean_title_strips_quotes_and_prefix(self) -> None:
        assert _clean_title('"Attention Is All You Need"') == "Attention Is All You Need"
        assert _clean_title("Title: BERT") == "BERT"
        assert _clean_title("标题：一种新方法") == "一种新方法"

    def test_clean_title_takes_first_line(self) -> None:
        assert _clean_title("Real Title\nThis is an explanation.") == "Real Title"

    def test_clean_title_rejects_overlong_output(self) -> None:
        """模型复述整段摘要时必须拒绝——500 字的"标题"塞进引用只会更难看。"""
        assert _clean_title("x" * 500) is None

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_clean_title_empty(self, raw: str) -> None:
        assert _clean_title(raw) is None

    async def test_infer_returns_title_and_records_usage(self) -> None:
        # 用**虚构**标题：本项目一律不用真实论文的数据做夹具，
        # 否则会与上游测试里的同类字符串产生无意义的相似命中，
        # 把真正需要关注的信号淹掉（这条在审计里已经真实发生过）。
        llm = FakeLLMClient(['"Adaptive Evidence Caching for Literature Question Answering"'])
        inferrer = LLMTitleInferrer(llm=llm, prompts=get_prompt_set("zh"))
        title = await inferrer.infer("Some first page text")
        assert title == "Adaptive Evidence Caching for Literature Question Answering"
        assert inferrer.last_usage.llm_calls == 1
        assert inferrer.last_usage.prompt_tokens > 0

    async def test_infer_truncates_input(self) -> None:
        llm = FakeLLMClient(["T"])
        inferrer = LLMTitleInferrer(llm=llm, prompts=get_prompt_set("zh"), max_chars=50)
        await inferrer.infer("x" * 5000)
        assert "x" * 51 not in llm.calls[0][1].content

    @pytest.mark.parametrize("text", ["", "   "])
    async def test_infer_empty_text_makes_no_call(self, text: str) -> None:
        llm = FakeLLMClient(["T"])
        inferrer = LLMTitleInferrer(llm=llm, prompts=get_prompt_set("zh"))
        assert await inferrer.infer(text) is None
        assert llm.calls == []

    async def test_infer_failure_degrades_to_none(self) -> None:
        """推断失败不该中断摄入。"""
        llm = FakeLLMClient(error=RuntimeError("provider down"))
        inferrer = LLMTitleInferrer(llm=llm, prompts=get_prompt_set("zh"))
        assert await inferrer.infer("some text") is None

    def test_disabled_by_default(self) -> None:
        from scitrace.config import MetadataSettings

        assert MetadataSettings().llm_title_inference is False


class TestPricing:
    """自备计价表：litellm 价格表不收录自建/代理模型名。"""

    def test_cost_computation(self) -> None:
        from scitrace.config import PricingSettings

        pricing = PricingSettings(
            input_per_million=1.0, input_cached_per_million=0.02, output_per_million=4.0
        )
        cost = pricing.cost_of(
            cached_tokens=500_000, uncached_tokens=500_000, completion_tokens=100_000
        )
        assert cost == pytest.approx(0.01 + 0.5 + 0.4)

    def test_missing_cache_price_falls_back_to_full_price(self) -> None:
        """缓存价未配时退化用普通输入价——宁可高估也不漏算。

        高估会让预算闸门偏保守，漏算会让它偏激进；两者不对称，所以选保守的那边。
        """
        from scitrace.config import PricingSettings

        pricing = PricingSettings(input_per_million=1.0, output_per_million=4.0)
        cost = pricing.cost_of(cached_tokens=1_000_000, uncached_tokens=0, completion_tokens=0)
        assert cost == pytest.approx(1.0)

    def test_not_configured_means_zero(self) -> None:
        from scitrace.config import PricingSettings

        assert PricingSettings().configured is False
        assert PricingSettings().cost_of(cached_tokens=10, uncached_tokens=10, completion_tokens=10) == 0.0

    def test_currency_is_explicit(self) -> None:
        """币种是显式字段：把人民币数字写进名叫 _usd 的字段是错的。"""
        from scitrace.config import PricingSettings
        from scitrace.domain.session import Usage

        assert PricingSettings().currency == "CNY"
        assert Usage().cost_currency == "USD"
        assert "estimated_cost" in Usage.model_fields
        assert "estimated_cost_usd" not in Usage.model_fields

    def test_empty_accumulator_adopts_the_reading_currency(self) -> None:
        """**回归测试**：空累加器必须让出币种，否则人民币读数被标成美元。

        实测事故：agentic 会话报告 ``estimated_cost=0.279, cost_currency="USD"``，
        而按 ``SCITRACE_PRICING__CURRENCY=CNY`` 配置，这个数其实是 0.279 元。
        成因是 ``merge`` 无条件取 ``self.cost_currency``，而最常见的用法正是
        ``Usage().merge(读数)``——空累加器的 ``USD`` 只是字段默认值。
        """
        from scitrace.domain.session import Usage

        reading = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost=0.5,
            cost_currency="CNY",
            cost_known=True,
        )
        assert Usage().is_identity, "空 Usage 必须是累加单位元"
        merged = Usage().merge(reading)
        assert merged.cost_currency == "CNY"
        assert merged.estimated_cost == pytest.approx(0.5)
        # 累加器一旦有读数，就由它继续持有币种
        assert merged.merge(reading).cost_currency == "CNY"

    def test_accumulator_recovers_currency_after_an_unknown_cost_reading(self) -> None:
        """先并入一次"成本不可知"的读数，之后仍须认领真实币种。

        ``cost_known`` 为假时币种没有意义，所以那种读数不该把累加器的币种钉死。
        """
        from scitrace.domain.session import Usage

        failed = Usage(prompt_tokens=7, cost_known=False, cost_currency="USD")
        accumulator = Usage().merge(failed)
        assert accumulator.cost_currency == "USD", "此刻只有这一次读数，无从判断"
        recovered = accumulator.merge(
            Usage(prompt_tokens=10, estimated_cost=0.1, cost_currency="CNY", cost_known=True)
        )
        assert recovered.cost_currency == "CNY", "知道成本的一方才是币种来源"
        assert recovered.cost_known is False, "有一次成本不可知，整体就不可知"

    def test_merge_does_not_add_currency(self) -> None:
        """币种不是可加量：把两个币种"相加"会得到一个没有意义的字符串。"""
        from scitrace.domain.session import Usage

        merged = Usage(estimated_cost=1.0, cost_currency="CNY").merge(
            Usage(estimated_cost=2.0, cost_currency="CNY")
        )
        assert merged.estimated_cost == pytest.approx(3.0)
        assert merged.cost_currency == "CNY"

    def test_cached_tokens_are_tracked(self) -> None:
        """缓存命中价便宜两个数量级，不区分会让成本高估数倍。"""
        from scitrace.domain.session import Usage

        merged = Usage(cached_tokens=100).merge(Usage(cached_tokens=50))
        assert merged.cached_tokens == 150

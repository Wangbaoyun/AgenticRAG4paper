"""测试答案合成与引用绑定。

``bind_citations`` 是"可溯源"承诺真正落地的地方，也是本文件的重心。
它被写成纯函数正是为了能在这里穷举各种畸形输出——这些分支在端到端测试里
既贵又不稳定（模型输出每次不同），而它们恰恰是实际会遇到的。
"""

from __future__ import annotations

import pytest
from fakes import FakeLLMClient
from scitrace.config import AnswerSettings
from scitrace.domain import Answer, Evidence, Source, make_evidence_key
from scitrace.pipeline.synthesis import (
    AnswerSynthesizer,
    bind_citations,
    count_dangling,
    extract_evidence_keys,
)
from scitrace.prompts import INSUFFICIENT_EVIDENCE, get_prompt_set


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


def evidence(
    fragment_id: str = "frag000",
    *,
    source_key: str = "src-1",
    summary: str = "该段给出了 85.2% 的精确率。",
    relevance: int = 9,
    citation: str = "",
    page_label: str = "pages 3-4",
) -> Evidence:
    key = make_evidence_key(source_key, fragment_id)
    return Evidence(
        key=key,
        source_key=source_key,
        fragment_id=fragment_id,
        summary=summary,
        relevance=relevance,
        citation=citation or f"(张三2024证据可溯 {page_label})",
        page_label=page_label,
    )


class TestExtractEvidenceKeys:
    def test_single_key(self) -> None:
        assert extract_evidence_keys("结论是 X (ev-1a2b3c4d)。") == ["ev-1a2b3c4d"]

    def test_order_is_first_appearance(self) -> None:
        """参考文献必须按引用先后排列，因此顺序不能丢。"""
        text = "先 ev-bbbbbbbb 后 ev-aaaaaaaa 再 ev-bbbbbbbb"
        assert extract_evidence_keys(text) == ["ev-bbbbbbbb", "ev-aaaaaaaa"]

    def test_multiple_in_one_group(self) -> None:
        assert extract_evidence_keys("(ev-11111111, ev-22222222)") == [
            "ev-11111111",
            "ev-22222222",
        ]

    @pytest.mark.parametrize(
        "text",
        ["没有引用键", "", "ev-1234567", "ev-zzzzzzzz", "EV-12345678", "ev-123456780"],
    )
    def test_non_keys_are_ignored(self, text: str) -> None:
        """只匹配 8 位十六进制的完整形态：``ev-`` 后面跟任意文本不是键。"""
        assert extract_evidence_keys(text) == []

    def test_dangling_count(self) -> None:
        items = [evidence("frag000")]
        text = f"有效 {items[0].key} 无效 ev-ffffffff"
        assert count_dangling(text, items) == 1


class TestBindCitations:
    def test_replaces_group_with_inline_citation(self) -> None:
        item = evidence()
        answer = bind_citations(
            f"精确率为 85.2% ({item.key})。", evidence=[item], sources={"src-1": source()}
        )
        assert answer.text == "精确率为 85.2% (张三2024证据可溯 pages 3-4)。"
        assert answer.refused is False

    def test_multiple_keys_in_one_group(self) -> None:
        a, b = evidence("frag000"), evidence("frag001", page_label="page 7")
        answer = bind_citations(
            f"两项研究一致 ({a.key}, {b.key})。", evidence=[a, b], sources={"src-1": source()}
        )
        assert answer.text == (
            "两项研究一致 (张三2024证据可溯 pages 3-4, 张三2024证据可溯 page 7)。"
        )

    @pytest.mark.parametrize("separator", ["; ", " and ", "-", ", "])
    def test_accepts_noncompliant_separators(self, separator: str) -> None:
        """提示词禁止分号/连词/连字符，但模型仍会写。

        格式不合规**不是丢弃引用的理由**——丢掉的是可溯源性，
        而那是本系统的核心承诺。
        """
        a, b = evidence("frag000"), evidence("frag001")
        answer = bind_citations(
            f"结论 ({a.key}{separator}{b.key})。", evidence=[a, b], sources={"src-1": source()}
        )
        assert answer.text.count("张三2024证据可溯") == 2
        assert len(answer.citations) == 2

    def test_bracket_form_from_material_is_handled(self) -> None:
        """模型会照抄材料里的 ``[ev-xxx]`` 方括号形式。"""
        item = evidence()
        answer = bind_citations(
            f"如材料所述 [{item.key}]，精确率为 85.2%。",
            evidence=[item],
            sources={"src-1": source()},
        )
        assert "[ev-" not in answer.text
        assert "(张三2024证据可溯 pages 3-4)" in answer.text

    def test_bare_key_without_parentheses(self) -> None:
        item = evidence()
        answer = bind_citations(
            f"结论见 {item.key}。", evidence=[item], sources={"src-1": source()}
        )
        assert answer.text == "结论见 (张三2024证据可溯 pages 3-4)。"

    def test_dangling_keys_are_stripped(self) -> None:
        """编造的键必须剔除。

        保留一个格式正确却指向不存在文献的引用**比没有引用更糟**——
        它看起来经过了核对，而读者不会去验证。
        """
        item = evidence()
        answer = bind_citations(
            f"真结论 ({item.key})。假结论 (ev-ffffffff)。",
            evidence=[item],
            sources={"src-1": source()},
        )
        assert "ev-ffffffff" not in answer.text
        assert "(ev-ffffffff)" not in answer.text
        assert len(answer.citations) == 1

    def test_group_with_only_dangling_keys_is_removed(self) -> None:
        item = evidence()
        answer = bind_citations(
            "无依据的结论 (ev-ffffffff)。", evidence=[item], sources={"src-1": source()}
        )
        assert "ev-" not in answer.text
        assert "()" not in answer.text

    def test_empty_parentheses_are_tidied(self) -> None:
        item = evidence()
        answer = bind_citations(f"结论 () 还有 ({item.key})。", evidence=[item], sources={})
        assert "()" not in answer.text

    def test_answer_without_citations_is_refused(self) -> None:
        """没有任何引用的答案不能当作正常结论呈现。"""
        item = evidence()
        answer = bind_citations("我认为答案是 42。", evidence=[item], sources={"src-1": source()})
        assert answer.refused is True
        assert "未引用" in answer.refusal_reason

    def test_refusal_sentinel_produces_refusal(self) -> None:
        item = evidence()
        answer = bind_citations(
            f"{INSUFFICIENT_EVIDENCE}", evidence=[item], sources={"src-1": source()}
        )
        assert answer.refused is True
        assert answer.citations == []
        assert answer.references == {}

    def test_refusal_strips_citations_even_if_present(self) -> None:
        """拒答**必须不带引用**，否则读者会以为它是基于证据得出的结论。"""
        item = evidence()
        answer = bind_citations(
            f"{INSUFFICIENT_EVIDENCE} 但也许可以参考 ({item.key})",
            evidence=[item],
            sources={"src-1": source()},
        )
        assert answer.refused is True
        assert answer.citations == []

    def test_chinese_refusal_is_recognized(self) -> None:
        item = evidence()
        answer = bind_citations("证据不足，无法回答", evidence=[item], sources={})
        assert answer.refused is True

    def test_long_answer_quoting_the_refusal_phrase_still_binds_citations(self) -> None:
        """**回归测试**：正文引用材料里的"证据不足"不得让整段答案被判为拒答。

        实测事故（agentic 模式，2024 语料真实问答）：模型产出了 2,228 字、
        结构完整、引用齐全的答案，只因正文中段（第 1,781 字符）引用了论文原文
        "若证据不足，可搜索更多论文…"，就被 ``is_refusal`` 的子串匹配判为拒答。
        后果不是"少答一题"，而是**把最好的答案扔掉了**：``refused=True``、
        ``citations == []``、引用绑定根本没执行，用户看到的是满屏未替换的
        ``ev-`` 裸键。

        拒答是一句**声明**（出现在开头），引用是**内容**（出现在中间）。
        """
        item = evidence()
        quotation = "若证据不足，可搜索更多论文、收集先前证据引用的论文、或换短语重新收集证据"
        head = "该 agent 的关键工具包括论文搜索、证据收集与答案生成三类，三者构成一个可迭代的检索闭环。"
        body = f"其中证据收集工具的设计动机是：{quotation}。" * 6
        answer = bind_citations(
            f"{head}{body}综上，这些工具共同构成 agentic 检索流程 (ev-"
            f"{item.key.removeprefix('ev-')})。",
            evidence=[item],
            sources={"src-1": source()},
        )
        assert len(answer.raw_text) > 120, "必须长于短答案阈值，否则测的是另一条分支"
        assert quotation in answer.raw_text, "被引用的原文必须真的在答案里"
        # 引文必须落在"开头窗口"之外，否则测的是另一条分支（短答案/开头声明）
        assert answer.raw_text.index(quotation) > 20, "引文位置太靠前，没测到长答案分支"
        assert answer.refused is False, "引用材料不是拒答"
        assert answer.citations, "引用必须被成功绑定"
        assert "ev-" not in answer.text, "裸键必须已被替换为可读引用"

    def test_references_are_in_first_citation_order(self) -> None:
        b = evidence("frag001", source_key="src-b")
        a = evidence("frag000", source_key="src-a")
        sources = {
            "src-a": source("src-a", title="Alpha Study of Retrieval"),
            "src-b": source("src-b", title="Beta Study of Retrieval"),
        }
        answer = bind_citations(
            f"先 {b.key} 后 {a.key}", evidence=[a, b], sources=sources
        )
        assert list(answer.references) == ["张三2024beta", "张三2024alpha"]

    def test_references_deduplicated_by_source(self) -> None:
        first = evidence("frag000")
        second = evidence("frag001", page_label="page 9")
        answer = bind_citations(
            f"A {first.key} B {second.key}", evidence=[first, second], sources={"src-1": source()}
        )
        assert len(answer.citations) == 2
        assert len(answer.references) == 1

    def test_missing_source_still_yields_reference_entry(self) -> None:
        """元数据缺失时仍生成条目，避免参考文献表与正文引用数量对不上。"""
        item = evidence()
        answer = bind_citations(f"结论 ({item.key})。", evidence=[item], sources={})
        assert "unknown" in answer.references
        assert "Metadata unavailable" in answer.references["unknown"]

    def test_raw_text_is_preserved(self) -> None:
        """保留原始输出以便排障与审计——改写后的文本无法反推模型到底说了什么。"""
        item = evidence()
        raw = f"结论 ({item.key})。"
        assert bind_citations(raw, evidence=[item], sources={}).raw_text == raw

    def test_bibtex_is_generated(self) -> None:
        item = evidence()
        answer = bind_citations(f"结论 ({item.key})。", evidence=[item], sources={"src-1": source()})
        entry = next(iter(answer.references.values()))
        assert entry.startswith("@")
        assert "证据可溯源问答方法研究" in entry


class TestAnswerSynthesizer:
    def make(self, llm: FakeLLMClient, **overrides) -> AnswerSynthesizer:
        return AnswerSynthesizer(
            llm=llm, prompts=get_prompt_set("zh"), settings=AnswerSettings(**overrides)
        )

    async def test_empty_evidence_refuses_without_calling_model(self) -> None:
        """空上下文下让模型作答，等于请它凭记忆编造。"""
        llm = FakeLLMClient()
        answer = await self.make(llm).synthesize("问题", [], sources={})
        assert answer.refused is True
        assert llm.calls == []

    async def test_context_contains_evidence_keys(self) -> None:
        item = evidence()
        llm = FakeLLMClient([f"结论 ({item.key})。"])
        await self.make(llm).synthesize("问题", [item], sources={"src-1": source()})

        _, user = llm.calls[0][0].content, llm.calls[0][1].content
        assert f"[{item.key}]" in user
        assert "85.2%" in user
        assert "问题" in user

    async def test_returns_bound_answer(self) -> None:
        item = evidence()
        llm = FakeLLMClient([f"精确率为 85.2% ({item.key})。"])
        answer = await self.make(llm).synthesize("问题", [item], sources={"src-1": source()})
        assert isinstance(answer, Answer)
        assert "(张三2024证据可溯 pages 3-4)" in answer.text
        assert answer.refused is False

    async def test_max_evidence_limits_context(self) -> None:
        items = [evidence(f"frag{i:03d}") for i in range(6)]
        llm = FakeLLMClient([f"结论 ({items[0].key})。"])
        await self.make(llm, max_evidence=2).synthesize("问题", items, sources={"src-1": source()})

        user = llm.calls[0][1].content
        assert f"[{items[0].key}]" in user
        assert f"[{items[4].key}]" not in user

    async def test_usage_is_recorded(self) -> None:
        item = evidence()
        llm = FakeLLMClient([f"结论 ({item.key})。"])
        synthesizer = self.make(llm)
        await synthesizer.synthesize("问题", [item], sources={"src-1": source()})
        assert synthesizer.last_usage.llm_calls == 1
        assert synthesizer.last_usage.prompt_tokens == 10
        assert synthesizer.last_usage.dangling_citations == 0

    async def test_dangling_citations_are_counted(self, caplog) -> None:
        item = evidence()
        llm = FakeLLMClient([f"结论 ({item.key}) 还有 (ev-ffffffff)。"])
        synthesizer = self.make(llm)
        await synthesizer.synthesize("问题", [item], sources={"src-1": source()})
        assert synthesizer.last_usage.dangling_citations == 1

    async def test_prior_answer_is_injected(self) -> None:
        item = evidence()
        llm = FakeLLMClient([f"修订结论 ({item.key})。"])
        await self.make(llm).synthesize(
            "问题", [item], sources={"src-1": source()}, prior_answer="旧答案正文"
        )
        assert "旧答案正文" in llm.calls[0][0].content

    async def test_refusal_from_model(self) -> None:
        item = evidence()
        llm = FakeLLMClient([INSUFFICIENT_EVIDENCE])
        answer = await self.make(llm).synthesize("问题", [item], sources={})
        assert answer.refused is True

    async def test_usage_resets_between_calls(self) -> None:
        item = evidence()
        llm = FakeLLMClient([f"结论 ({item.key})。", f"结论 ({item.key})。"])
        synthesizer = self.make(llm)
        await synthesizer.synthesize("问题", [item], sources={"src-1": source()})
        await synthesizer.synthesize("问题", [item], sources={"src-1": source()})
        assert synthesizer.last_usage.llm_calls == 1

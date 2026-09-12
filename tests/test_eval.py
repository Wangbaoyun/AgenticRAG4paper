"""测试评测骨架。

评测工具本身也需要被测试——一个把指标算错的评测工具比没有评测更糟：
它会给出一个看起来可信的数字，而所有后续决策都建立在它之上。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scitrace.agent.runtime import AgentRunResult
from scitrace.domain import Answer
from scitrace.domain.session import SessionStatus, Usage

from benchmarks.qa_eval import (
    CaseResult,
    EvaluationCase,
    EvaluationReport,
    format_report,
    load_cases,
)


class TestLoadCases:
    def test_reads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "q1", "question": "问题一", "expected_keywords": ["85.2%"]}),
                    json.dumps({"id": "q2", "question": "问题二", "expected_answerable": False}),
                ]
            ),
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert [case.identifier for case in cases] == ["q1", "q2"]
        assert cases[0].expected_keywords == ("85.2%",)
        assert cases[1].expected_answerable is False

    def test_skips_comments_and_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text('# 注释\n\n{"question": "q"}\n', encoding="utf-8")
        assert len(load_cases(path)) == 1

    def test_malformed_line_does_not_break_the_set(self, tmp_path: Path) -> None:
        """200 题的评测集因一行多个逗号而完全跑不起来，比丢掉一道题严重得多。"""
        path = tmp_path / "c.jsonl"
        path.write_text('{not json}\n{"question": "ok"}\n', encoding="utf-8")
        cases = load_cases(path)
        assert len(cases) == 1

    def test_missing_question_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        path.write_text('{"id": "x"}\n', encoding="utf-8")
        assert load_cases(path) == []


def result(**overrides) -> CaseResult:
    base = {
        "case": EvaluationCase(question="q", expected_answerable=True),
        "status": "SUCCESS",
        "refused": False,
        "answer": "答案是 85.2%",
        "citation_count": 1,
        "dangling_citations": 0,
        "duration_s": 1.0,
        "cost": 0.01,
        "currency": "CNY",
    }
    base.update(overrides)
    return CaseResult(**base)


class TestMetrics:
    def test_refusal_correctness_is_the_key_metric(self) -> None:
        """一个"什么都敢答"的系统在只测可回答问题的评测里会拿满分。

        因此拒答题目的行为必须单独计入。
        """
        report = EvaluationReport(
            results=[
                result(case=EvaluationCase(question="可答", expected_answerable=True)),
                result(
                    case=EvaluationCase(question="不可答", expected_answerable=False),
                    refused=False,  # 该拒没拒
                ),
            ]
        )
        assert report.behavior_correctness == pytest.approx(0.5)

    def test_correct_refusal_counts_as_correct(self) -> None:
        report = EvaluationReport(
            results=[
                result(
                    case=EvaluationCase(question="库外", expected_answerable=False), refused=True
                )
            ]
        )
        assert report.behavior_correctness == pytest.approx(1.0)

    def test_keyword_coverage_only_counts_answerable(self) -> None:
        report = EvaluationReport(
            results=[
                result(case=EvaluationCase(question="a", expected_keywords=("85.2%",))),
                result(
                    case=EvaluationCase(question="b", expected_keywords=("nope",)),
                    answer="别的内容",
                ),
            ]
        )
        assert report.answer_coverage == pytest.approx(0.5)

    def test_no_keywords_means_hit(self) -> None:
        assert result().keyword_hit is True

    def test_citation_resolution_rate_penalizes_dangling(self) -> None:
        report = EvaluationReport(
            results=[result(citation_count=3, dangling_citations=1)]
        )
        assert report.citation_resolution_rate == pytest.approx(0.75)

    def test_perfect_citation_rate(self) -> None:
        report = EvaluationReport(results=[result(citation_count=2, dangling_citations=0)])
        assert report.citation_resolution_rate == pytest.approx(1.0)

    def test_no_citations_is_treated_as_perfect(self) -> None:
        """拒答不带引用是正确行为，不该被算成引用解析失败。"""
        report = EvaluationReport(
            results=[result(citation_count=0, dangling_citations=0, refused=True)]
        )
        assert report.citation_resolution_rate == pytest.approx(1.0)

    def test_hallucination_rate(self) -> None:
        report = EvaluationReport(
            results=[
                result(
                    case=EvaluationCase(question="a", forbidden_keywords=("大概",)),
                    answer="大概是 85.2%",
                ),
                result(),
            ]
        )
        assert report.hallucination_rate == pytest.approx(0.5)

    def test_cost_and_latency_averages(self) -> None:
        report = EvaluationReport(
            results=[result(cost=0.02, duration_s=2.0), result(cost=0.04, duration_s=4.0)]
        )
        assert report.cost_per_question == pytest.approx(0.03)
        assert report.mean_latency_s == pytest.approx(3.0)

    def test_empty_report_is_all_zero(self) -> None:
        report = EvaluationReport()
        assert report.total == 0
        assert report.behavior_correctness == 0.0
        assert report.citation_resolution_rate == pytest.approx(1.0)

    def test_summary_is_json_serializable(self) -> None:
        payload = EvaluationReport(results=[result()]).summary()
        assert json.dumps(payload)
        assert "behavior_correctness" in payload


class TestFormatReport:
    def test_renders_markdown(self) -> None:
        rendered = format_report(EvaluationReport(results=[result()]))
        assert rendered.startswith("# scitrace 评测报告")
        assert "behavior_correctness" in rendered
        assert "| # |" in rendered


class TestCurrencyPropagation:
    """币种必须一路传到报告顶层。

    真实评测里出现过一次：成本值是对的（21.5 元），但报告写的是 `cost_currency: USD`
    ——因为 `CaseResult` 的默认值是 USD，而成功路径忘了把它传进去。
    读数的人会以为这是美元，差了一个数量级的判断都可能因此成立。
    """

    def test_report_uses_case_currency(self) -> None:
        report = EvaluationReport(results=[result(currency="CNY")])
        assert report.cost_currency == "CNY"
        assert report.summary()["cost_currency"] == "CNY"

    def test_defaults_to_usd_when_empty(self) -> None:
        assert EvaluationReport().cost_currency == "USD"

    def test_summary_is_serializable_with_currency(self) -> None:
        payload = EvaluationReport(results=[result(currency="CNY")]).summary()
        assert json.loads(json.dumps(payload))["cost_currency"] == "CNY"

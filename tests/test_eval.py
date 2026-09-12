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


class TestRepeatAndSpread:
    """运行间方差必须被度量出来，否则同一配置的两次运行会被当成两个配置的差异。

    实测背景：agentic 模式同一道题、同一索引，LLM 调用次数在 30–87 之间浮动
    （见 ``docs/EXPERIMENTS.md`` 实验六 6.2）。只跑一次的对照拿到的是随机落点。
    """

    def _cases(self) -> list[EvaluationCase]:
        return [
            EvaluationCase(identifier="a01", question="q1", expected_answerable=True),
            EvaluationCase(identifier="a02", question="q2", expected_answerable=True),
        ]

    async def test_repeat_runs_each_case_the_requested_number_of_times(self) -> None:
        from benchmarks.qa_eval import run_evaluation

        calls: list[str] = []

        class _Services:
            class settings:  # noqa: N801
                class pricing:  # noqa: N801
                    currency = "CNY"

        async def fake_ask(question, services, *, mode="deterministic"):  # noqa: ANN001, ARG001
            calls.append(question)
            return AgentRunResult(
                question=question,
                answer=Answer(text="答案", raw_text="答案", citations=[]),
                status=SessionStatus.SUCCESS,
                usage=Usage(prompt_tokens=10, completion_tokens=5, llm_calls=2),
            )

        import scitrace.api as api

        original_api = api.ask
        api.ask = fake_ask  # type: ignore[assignment]
        try:
            # run_evaluation 在调用时 `from scitrace.api import ask`，
            # 所以替换模块属性即可生效。
            report = await run_evaluation(self._cases(), _Services(), mode="agentic", repeat=3)
        finally:
            api.ask = original_api  # type: ignore[assignment]

        assert calls == ["q1", "q2"] * 3, "应按轮次重复整批，而不是逐题连跑"
        assert report.repeats == 3
        assert [item.run_index for item in report.results] == [0, 0, 1, 1, 2, 2]

    def test_spread_reports_min_median_max_per_case(self) -> None:
        report = EvaluationReport(
            results=[
                result(case=EvaluationCase(identifier="a01", question="q"), llm_calls=30),
                result(case=EvaluationCase(identifier="a01", question="q"), llm_calls=87),
                result(case=EvaluationCase(identifier="a01", question="q"), llm_calls=50),
                result(case=EvaluationCase(identifier="a02", question="q"), llm_calls=12),
                result(case=EvaluationCase(identifier="a02", question="q"), llm_calls=14),
            ]
        )
        rows = dict((row[0], row[1:]) for row in report.per_case_spread())
        assert rows["a01"] == (30, 50, 87), "中位数不是平均值，别把极值拉进来"
        assert rows["a02"] == (12, 14, 14)

    def test_cost_spread_ratio_uses_per_case_median(self) -> None:
        """逐题取最贵/最便宜，再取中位数——避免被单题异常值主导。"""
        report = EvaluationReport(
            results=[
                result(case=EvaluationCase(identifier="a01", question="q"), cost=0.1),
                result(case=EvaluationCase(identifier="a01", question="q"), cost=0.3),
                result(case=EvaluationCase(identifier="a02", question="q"), cost=0.2),
                result(case=EvaluationCase(identifier="a02", question="q"), cost=0.4),
                result(case=EvaluationCase(identifier="a03", question="q"), cost=1.0),
                result(case=EvaluationCase(identifier="a03", question="q"), cost=50.0),
            ]
        )
        # 逐题极差为 3.0 / 2.0 / 50.0，中位数是 3.0
        assert report.cost_spread_ratio == pytest.approx(3.0)

    def test_single_run_reports_no_spread(self) -> None:
        report = EvaluationReport(
            results=[result(case=EvaluationCase(identifier="a01", question="q"))]
        )
        assert report.repeats == 1
        assert report.cost_spread_ratio == pytest.approx(1.0)

    def test_markdown_warns_that_spread_precedes_the_summary(self) -> None:
        report = EvaluationReport(
            results=[
                result(case=EvaluationCase(identifier="a01", question="q"), llm_calls=30),
                result(case=EvaluationCase(identifier="a01", question="q"), llm_calls=87),
            ]
        )
        rendered = format_report(report)
        assert "运行间稳定性" in rendered
        assert "| a01 | 30 | 87 | 87 |" in rendered
        assert "先看这张表再看汇总" in rendered

    def test_error_fallback_uses_the_configured_currency(self) -> None:
        """失败题的币种不能写死 USD，否则人民币计价下的异常读数会被伪装成正常。"""
        source = (Path(__file__).resolve().parents[1] / "benchmarks" / "qa_eval.py").read_text(
            encoding="utf-8"
        )
        assert 'fallback_currency = services.settings.pricing.currency' in source


class TestHallucinationMetricIsNotVacuous:
    """``hallucination_rate`` 一度在整个评测集上**结构性恒为 0**。

    36 道题里没有任何一题定义 ``forbidden_keywords``，
    于是 ``forbidden_hit`` 恒为 False、这个指标无法取到非零值。
    它却出现在每一张实验表里被当作正面结果——正是项目自己在
    EXPERIMENTS.md 实验五里警告过的"指标饱和不是表现好，是测量没有区分度"。
    """

    def test_the_eval_set_actually_defines_forbidden_keywords(self) -> None:
        """判据存在性检查：否则指标恒为 0 而看不出来。"""
        cases = load_cases(Path(__file__).resolve().parents[1] / "benchmarks/data/rag_qa.jsonl")
        guarded = [case for case in cases if case.forbidden_keywords]
        assert guarded, "评测集没有禁止词 → hallucination_rate 结构性恒为 0"
        # 禁止词应当落在库外题上：那些题"编造答案"才是真正的幻觉
        assert all(not case.expected_answerable for case in guarded)
        assert len(guarded) >= 8

    def test_fabricated_answer_on_an_out_of_corpus_case_is_caught(self) -> None:
        case = EvaluationCase(
            identifier="n06",
            question="How many parameters does GPT-4 have?",
            expected_answerable=False,
            forbidden_keywords=("1.8 trillion",),
        )
        fabricated = result(case=case, answer="GPT-4 has about 1.8 trillion parameters.", refused=False)
        assert fabricated.forbidden_hit is True
        assert fabricated.behavior_correct is False
        assert EvaluationReport(results=[fabricated]).hallucination_rate == pytest.approx(1.0)

    def test_a_refusal_mentioning_the_number_is_not_a_hallucination(self) -> None:
        """拒答没有主张内容；把拒答算成幻觉会让"该拒的拒了"反受惩罚。"""
        case = EvaluationCase(
            identifier="n06",
            question="How many parameters does GPT-4 have?",
            expected_answerable=False,
            forbidden_keywords=("1.8 trillion",),
        )
        refused = result(
            case=case,
            answer="语料未涉及 GPT-4 的参数量（坊间流传的 1.8 trillion 无出处）。",
            refused=True,
        )
        assert refused.forbidden_hit is False
        assert refused.behavior_correct is True
        assert EvaluationReport(results=[refused]).hallucination_rate == pytest.approx(0.0)

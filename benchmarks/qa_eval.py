"""自主评测骨架：【原创增量 ⑤ 的落点】

SPEC §9 把评测的**执行**列为 M8（超出首版范围），但把**落点**放在这里。
一个空的 ``benchmarks/`` 目录只是承诺，而一个能跑通的骨架是证据：
它证明前四项增量不是孤立的技巧，而是可以被同一套指标衡量的。

## 为什么评测指标必须由本项目自己定义

直接借用某个公开基准的分数没有意义：那些基准的语料、题型与拒绝回答的判定方式
都与本项目的目标不完全一致，用它的分数只能说明"在别的任务上表现如何"。
因此这里定义的是**与 SPEC §1.3 质量目标一一对应**的指标：

| 指标 | 对应 SPEC 质量目标 |
| --- | --- |
| ``citation_resolution_rate`` | 引用可回溯率 100% |
| ``refusal_correctness`` | 拒答正确性 |
| ``answer_coverage`` | —— |
| ``cost_per_question`` | 成本可观测 |

## 语料与题目的来源

评测集是 **JSONL**，一行一题。它**刻意不从任何公开基准复制**：
LitQA2 一类的数据集有自己的许可，搬运它会让"零上游复制"的主张多出一个
需要单独解释的例外。自建集虽然规模小，但它的每一道题都是为本项目的语料
量身写的——这恰恰是"评测"这个词应有的含义。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from scitrace.prompts import is_refusal

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "CaseResult",
    "load_cases",
    "run_evaluation",
]


@dataclass(frozen=True)
class EvaluationCase:
    """一道评测题。

    ``expected_answerable=False`` 的题**尤其是**有价值的：拒答能力无法从
    "答对率"里看出来，必须专门构造"库里根本没有答案"的问题去测。
    只测可回答问题的评测会让一个"什么都敢答"的系统拿到满分。
    """

    question: str
    expected_answerable: bool = True
    #: 答案中应当出现的子串（不区分大小写）。留空表示只检查结构性质。
    expected_keywords: tuple[str, ...] = ()
    #: 答案中**不应**出现的子串，用于检测典型的幻觉表述。
    forbidden_keywords: tuple[str, ...] = ()
    identifier: str = ""


@dataclass
class CaseResult:
    """单题结果。"""

    case: EvaluationCase
    status: str
    refused: bool
    answer: str
    citation_count: int
    dangling_citations: int
    duration_s: float
    cost: float
    currency: str = "USD"
    #: 该题是第几次重复（从 0 开始）。不重复时恒为 0。
    run_index: int = 0
    #: 该题的 LLM 调用次数与 token 总量——**方差的主要来源**。
    llm_calls: int = 0
    tokens: int = 0

    @property
    def keyword_hit(self) -> bool:
        """期望关键词是否全部命中（无期望关键词时恒为 True）。"""
        if not self.case.expected_keywords:
            return True
        lowered = self.answer.lower()
        return all(word.lower() in lowered for word in self.case.expected_keywords)

    @property
    def forbidden_hit(self) -> bool:
        """是否出现了禁止词（无禁止词时恒为 False）。

        **只对非拒答的答案计。** 拒答没有主张任何内容——即便它在解释里
        顺带提到那个数字（"本语料未涉及 GPT-4 参数量，只有 GPT-3 的 1750 亿"），
        也不构成"给出了不该给的答案"。把拒答算成幻觉会让"该拒的拒了"
        反而受到惩罚，方向正好反了。
        """
        if self.refused:
            return False
        lowered = self.answer.lower()
        return any(word.lower() in lowered for word in self.case.forbidden_keywords)

    @property
    def behavior_correct(self) -> bool:
        """行为是否正确：该答的答了、该拒的拒了。

        这是比"答对率"更根本的指标——一个把库外问题也强行作答的系统，
        无论答对多少题都不可信。
        """
        if not self.case.expected_answerable:
            return self.refused
        return not self.refused


@dataclass
class EvaluationReport:
    """一次评测的汇总。"""

    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def behavior_correctness(self) -> float:
        """行为正确率：该答的答、该拒的拒。"""
        if not self.results:
            return 0.0
        return sum(item.behavior_correct for item in self.results) / self.total

    @property
    def answer_coverage(self) -> float:
        """可回答问题中，关键词全部命中的比例。"""
        answerable = [item for item in self.results if item.case.expected_answerable]
        if not answerable:
            return 0.0
        return sum(item.keyword_hit for item in answerable) / len(answerable)

    @property
    def citation_resolution_rate(self) -> float:
        """引用的可回溯率：答案中出现的引用键有多少能解析到真实证据。

        SPEC §1.3 要求 100%。低于 100% 意味着答案里有"看起来像引用、
        却指向不存在文献"的键——这是最需要警惕的失效模式。
        """
        total = sum(item.citation_count + item.dangling_citations for item in self.results)
        if total == 0:
            return 1.0
        return sum(item.citation_count for item in self.results) / total

    @property
    def hallucination_rate(self) -> float:
        """命中禁止词的比例，用于观察典型的无据表述。"""
        if not self.results:
            return 0.0
        return sum(item.forbidden_hit for item in self.results) / self.total

    @property
    def repeats(self) -> int:
        """每题重复了几次（未重复时为 1）。"""
        if not self.results:
            return 1
        counts: dict[str, int] = {}
        for item in self.results:
            counts[item.case.identifier] = counts.get(item.case.identifier, 0) + 1
        return max(counts.values())

    def per_case_spread(self) -> list[tuple[str, int, int, int]]:
        """每题的 LLM 调用次数分布：``(题号, 最小, 中位, 最大)``。

        **这是本评测最重要的一张表。**同一道题、同一份索引、同一套配置，
        agentic 模式的 LLM 调用次数实测可以在 30 到 87 之间浮动
        （见 ``docs/EXPERIMENTS.md`` 实验六 6.2）。只跑一次的对照
        得到的是"这一次的随机落点"，不是配置之间的差异——
        把它当作结论会得出**符号都可能反的**判断。
        """
        grouped: dict[str, list[int]] = {}
        for item in self.results:
            grouped.setdefault(item.case.identifier, []).append(item.llm_calls)
        rows = []
        for identifier, values in grouped.items():
            ordered = sorted(values)
            rows.append(
                (identifier, ordered[0], ordered[len(ordered) // 2], ordered[-1])
            )
        return rows

    @property
    def cost_spread_ratio(self) -> float:
        """最贵一次 / 最便宜一次的每题成本之比，用于度量运行间不稳定程度。

        逐题取"该题各次成本的最大值 / 最小值"，再取所有题的中位数，
        避免被单题异常值主导。1.0 表示完全稳定。
        """
        grouped: dict[str, list[float]] = {}
        for item in self.results:
            grouped.setdefault(item.case.identifier, []).append(item.cost)
        ratios = []
        for values in grouped.values():
            positive = [value for value in values if value > 0]
            if len(positive) >= 2:
                ratios.append(max(positive) / min(positive))
        if not ratios:
            return 1.0
        ratios.sort()
        return ratios[len(ratios) // 2]

    @property
    def cost_per_question(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.cost for item in self.results) / self.total

    @property
    def tokens_per_question(self) -> float:
        """每题 token 数。

        **跨配置对比一律看这个，不看金额。** 金额会被计价表、币种、缓存命中价、
        计价表缺项（litellm 不收录自建模型名）任意一层弄错——本项目已经因此
        先后出过"人民币标成美元""成本恒为 0 导致闸门失效""累计值当每题值"
        三类错误。token 直接来自提供商返回的 usage，没有中间解释层。
        """
        if not self.results:
            return 0.0
        return sum(item.tokens for item in self.results) / self.total

    @property
    def cost_currency(self) -> str:
        """本次评测的计价币种（取首题结果；空报告时为 USD）。"""
        return self.results[0].currency if self.results else "USD"

    @property
    def mean_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return sum(item.duration_s for item in self.results) / self.total

    def summary(self) -> dict[str, object]:
        """汇总为可直接写入结果文件的字典。"""
        return {
            "total": self.total,
            "behavior_correctness": round(self.behavior_correctness, 4),
            "answer_coverage": round(self.answer_coverage, 4),
            "citation_resolution_rate": round(self.citation_resolution_rate, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "tokens_per_question": round(self.tokens_per_question, 1),
            "cost_per_question": round(self.cost_per_question, 6),
            "cost_currency": self.cost_currency,
            "mean_latency_s": round(self.mean_latency_s, 3),
        }


def load_cases(path: Path) -> list[EvaluationCase]:
    """从 JSONL 读取评测集。

    单行损坏时跳过并警告，而不是让整个评测集不可用——一份 200 题的评测集
    因为一行多了个逗号而完全跑不起来，是远比丢掉一道题更严重的故障。
    """
    cases: list[EvaluationCase] = []
    malformed = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
            cases.append(
                EvaluationCase(
                    question=payload["question"],
                    expected_answerable=bool(payload.get("expected_answerable", True)),
                    expected_keywords=tuple(payload.get("expected_keywords", ())),
                    forbidden_keywords=tuple(payload.get("forbidden_keywords", ())),
                    identifier=str(payload.get("id", f"line-{line_number}")),
                )
            )
        except (KeyError, ValueError, TypeError) as error:
            malformed += 1
            logger.warning("评测集第 %d 行无法解析，已跳过：%s", line_number, error)
    if malformed:
        logger.warning("评测集共有 %d 行被跳过", malformed)
    return cases


async def run_evaluation(
    cases: Sequence[EvaluationCase],
    services,  # noqa: ANN001
    *,
    mode: str = "deterministic",
    repeat: int = 1,
) -> EvaluationReport:
    """把评测集跑一遍。

    刻意**顺序执行**而非并发：并发会让成本与延迟的统计失去意义
    （它们本是用于横向对比不同配置的指标），也让失败题目的复现变难。

    Args:
        cases: 评测题。
        services: 已装配的服务。
        mode: ``agentic`` 或 ``deterministic``。
        repeat: 每题重复次数。agentic 模式**必须**大于 1 才有对照价值——
            同一道题的 LLM 调用次数实测可在 30–87 之间浮动，
            单次运行只是随机落点，不是配置差异。

    Returns:
        汇总报告；``repeat > 1`` 时每题会有 ``repeat`` 条结果，用 ``run_index`` 区分。
    """
    from scitrace.api import ask  # noqa: PLC0415 - 避免评测骨架被生产路径导入

    fallback_currency = services.settings.pricing.currency
    report = EvaluationReport()
    for run_index in range(max(1, repeat)):
        for case in cases:
            started = time.perf_counter()
            try:
                result = await ask(case.question, services, mode=mode)
                status = str(result.status)
                answer = result.answer.text
                refused = result.answer.refused or is_refusal(answer)
                citations = len(result.answer.citations)
                dangling = result.usage.dangling_citations
                cost = result.usage.estimated_cost
                currency = result.usage.cost_currency
                llm_calls = result.usage.llm_calls
                tokens = result.usage.total_tokens
            except Exception as error:  # noqa: BLE001 - 单题失败不应中断整个评测
                logger.exception("评测题 %s 执行失败", case.identifier)
                status, answer, refused, citations, dangling, cost, currency = (
                    "ERROR",
                    f"执行失败：{error}",
                    True,
                    0,
                    0,
                    0.0,
                    # 用配置里的币种，不要写死 USD：写死会让"人民币计价下
                    # 成本恒为 0"这类问题在报告里伪装成正常读数。
                    fallback_currency,
                )
                llm_calls = tokens = 0
            report.results.append(
                CaseResult(
                    case=case,
                    status=status,
                    refused=refused,
                    answer=answer,
                    citation_count=citations,
                    dangling_citations=dangling,
                    duration_s=time.perf_counter() - started,
                    cost=cost,
                    currency=currency,
                    run_index=run_index,
                    llm_calls=llm_calls,
                    tokens=tokens,
                )
            )
            logger.info(
                "[%s#%d] %s → %s", case.identifier, run_index, case.question[:40], status
            )
    return report


def format_report(report: EvaluationReport) -> str:
    """把评测报告渲染为 Markdown，便于归档与对比。"""
    lines = ["# scitrace 评测报告", "", "## 汇总", ""]
    for key, value in report.summary().items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## 逐题结果", "", "| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for index, item in enumerate(report.results, start=1):
        tag = f"{item.case.identifier}#{item.run_index}" if report.repeats > 1 else str(index)
        lines.append(
            f"| {tag} | {item.case.question[:40]} | {item.status} | "
            f"{'是' if item.refused else '否'} | {item.citation_count} | "
            f"{'命中' if item.keyword_hit else '未命中'} | {item.duration_s:.1f}s |"
        )
    if report.repeats > 1:
        lines.extend(
            [
                "",
                "## 运行间稳定性（**先看这张表再看汇总**）",
                "",
                f"- 每题重复次数：`{report.repeats}`",
                f"- 逐题成本极差中位数（最贵/最便宜）：`{report.cost_spread_ratio:.2f}×`",
                "",
                "| 题号 | LLM 调用 最少 | 中位 | 最多 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for identifier, low, mid, high in report.per_case_spread():
            lines.append(f"| {identifier} | {low} | {mid} | {high} |")
        lines.extend(
            [
                "",
                "> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；"
                "拿不同配置的单次运行互比会得到符号都可能相反的结论。",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    """``python -m benchmarks.qa_eval --cases <jsonl> --settings <name>``。"""
    import argparse

    parser = argparse.ArgumentParser(description="运行 scitrace 评测集")
    parser.add_argument("--cases", type=Path, required=True, help="评测集 JSONL")
    parser.add_argument("--settings", default=None)
    parser.add_argument("--mode", choices=["agentic", "deterministic"], default="deterministic")
    parser.add_argument(
        "--ids",
        default=None,
        help="只跑这些题号，逗号分隔（如 a01,a05,s02）。默认跑全部。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "每题重复次数。agentic 模式务必大于 1：同题 LLM 调用次数实测可浮动 30–87，"
            "单次运行只是随机落点。"
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Markdown 报告输出路径")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from scitrace.api import load_services
    from scitrace.config import load_settings

    cases = load_cases(args.cases)
    if args.ids:
        wanted = [item.strip() for item in args.ids.split(",") if item.strip()]
        by_id = {case.identifier: case for case in cases}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            parser.error(f"评测集里没有这些题号：{', '.join(missing)}")
        cases = [by_id[item] for item in wanted]

    services = load_services(load_settings(name=args.settings))
    try:
        report = anyio.run(
            lambda: run_evaluation(
                cases, services, mode=args.mode, repeat=args.repeat
            )
        )
    finally:
        anyio.run(services.aclose)

    rendered = format_report(report)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

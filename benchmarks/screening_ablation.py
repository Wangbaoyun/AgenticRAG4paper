"""检索策略 × 筛选后端的完整对照实验（4 × 2 = 8 组配置）。

## 为什么两个轴要一起做

检索与筛选是串联的：召回质量决定筛选的上限，筛选决定送进合成的材料质量。
只测一个轴会把另一个轴的短板算到它头上。例如"cross_encoder 筛选更省 token"
这个结论在弱召回下可能完全不成立——因为候选本身就少。

## 判定标准

- `behavior_correctness`：该答的答、该拒的拒（**最重要的指标**）；
- `answer_coverage`：可回答问题中关键词命中率；
- `citation_resolution_rate`：引用可回溯率（SPEC 要求 100%）；
- 成本（元）与延迟——决定哪个配置值得作为默认。

## 规模

5 道可答题 + 2 道语料外题 × 8 组 = 56 次问答。
每次问答会真实调用 LLM，成本按 settings.pricing 计入。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.ERROR)

CASES = [
    ("What two memory types does the original RAG model combine?", True, ["parametric", "non-parametric"]),
    ("How does Corrective RAG handle retrieved documents that are irrelevant?", True, ["correct"]),
    ("What does the ReAct approach interleave?", True, ["reason"]),
    ("How does MemGPT manage context like an operating system?", True, ["memory"]),
    ("What reflection tokens does Self-RAG use?", True, ["token"]),
    ("What is the superconducting transition temperature of hydrogen sulfide at 150 GPa?", False, []),
    ("Who won the 2018 FIFA World Cup?", False, []),
]
STRATEGIES = ["dense", "dense_mmr", "hybrid_rrf", "hybrid_rrf_rerank"]
BACKENDS = ["llm", "cross_encoder"]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/ablation/screening_grid.json"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    from scitrace.api import ask
    from scitrace.config import load_settings
    from scitrace.factory import build_services

    base = load_settings()
    rows: list[dict] = []
    print(f"{'检索策略':<20}{'筛选后端':<15}{'行为正确':>9}{'关键词':>8}{'引用率':>8}{'成本(元)':>10}{'延迟':>9}")
    for strategy in STRATEGIES:
        for backend in BACKENDS:
            settings = load_settings()
            settings.retrieval.strategy = strategy          # type: ignore[assignment]
            settings.retrieval.k = args.k
            settings.screening.backend = backend            # type: ignore[assignment]
            services = build_services(settings, load_index=True)
            try:
                correct = covered = 0; answerable = 0
                citations = dangling = 0; cost = 0.0; durations = []; tokens = 0
                for question, expect_answerable, keywords in CASES:
                    started = time.perf_counter()
                    try:
                        result = await ask(question, services, mode="deterministic")
                        refused = result.answer.refused
                        answered_ok = (not refused) if expect_answerable else refused
                        correct += answered_ok
                        if expect_answerable:
                            answerable += 1
                            lowered = result.answer.text.lower()
                            covered += all(k in lowered for k in keywords)
                        citations += len(result.answer.citations)
                        dangling += result.usage.dangling_citations
                        cost += result.usage.estimated_cost
                        tokens += result.usage.total_tokens
                    except Exception as error:  # noqa: BLE001
                        logging.exception("题目失败：%s", question[:40])
                    durations.append(time.perf_counter() - started)
                row = {
                    "strategy": strategy, "backend": backend,
                    "behavior_correctness": round(correct / len(CASES), 3),
                    "answer_coverage": round(covered / answerable, 3) if answerable else 0.0,
                    "citation_resolution_rate": round(citations / (citations + dangling), 3)
                    if (citations + dangling) else 1.0,
                    "cost_cny": round(cost, 4), "tokens": tokens,
                    "mean_latency_s": round(sum(durations) / len(durations), 1),
                }
                rows.append(row)
                print(f"{strategy:<20}{backend:<15}{row['behavior_correctness']:>8.2f}"
                      f"{row['answer_coverage']:>8.2f}{row['citation_resolution_rate']:>8.2f}"
                      f"{row['cost_cny']:>10.4f}{row['mean_latency_s']:>8.1f}s", flush=True)
            finally:
                await services.aclose()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(r["cost_cny"] for r in rows)
    print(f"\n合计成本 {total:.4f} 元 | 报告 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

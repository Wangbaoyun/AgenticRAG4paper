# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.963
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.100769
- `cost_currency`: USD
- `mean_latency_s`: 37.371

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 30.7s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 42.5s |
| a23#0 | Besides vector databases, what other sou | TRUNCATED | 否 | 9 | 命中 | 71.7s |
| a24#0 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 22.2s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 4 | 命中 | 43.1s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 41.0s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 10 | 命中 | 84.1s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.3s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 23.0s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 12.7s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 12.1s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 47.4s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 31.0s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 否 | 4 | 命中 | 86.6s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 32.1s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 30.7s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 22.3s |
| n06#1 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 29.0s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 11.6s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 13.2s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 50.9s |
| a24#2 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 29.7s |
| s01#2 | Several papers here propose methods that | TRUNCATED | 否 | 5 | 未命中 | 45.0s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 8 | 命中 | 60.5s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 9 | 命中 | 83.8s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 12.2s |
| n06#2 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 28.7s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.92×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 15 | 24 |
| a05 | 15 | 15 | 15 |
| a23 | 41 | 44 | 53 |
| a24 | 32 | 33 | 44 |
| s01 | 35 | 37 | 64 |
| s02 | 25 | 25 | 37 |
| s03 | 43 | 49 | 76 |
| n04 | 22 | 23 | 24 |
| n06 | 34 | 44 | 44 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

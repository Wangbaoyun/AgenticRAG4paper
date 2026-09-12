# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.9259
- `answer_coverage`: 0.9048
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.062057
- `cost_currency`: CNY
- `mean_latency_s`: 34.622

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 1 | 命中 | 24.5s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 1 | 命中 | 9.6s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 6 | 命中 | 41.0s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 1 | 命中 | 28.9s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 6 | 命中 | 51.9s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 3 | 命中 | 25.5s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 10 | 命中 | 94.0s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 8.1s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 15.2s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 3 | 命中 | 7.9s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 1 | 命中 | 7.8s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 6 | 命中 | 87.6s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 18.4s |
| s01#1 | Several papers here propose methods that | SUCCESS | 否 | 6 | 命中 | 43.9s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 41.6s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 26.9s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.2s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 16.8s |
| a01#2 | How many parameters does GPT-3 have? | TRUNCATED | 否 | 1 | 命中 | 30.2s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 1 | 命中 | 9.1s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 6 | 命中 | 44.3s |
| a24#2 | Which organization is credited with intr | TRUNCATED | 是 | 0 | 未命中 | 21.7s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 65.8s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 76.8s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 8 | 命中 | 100.2s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 14.3s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 11.6s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`2.03×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 10 | 10 | 27 |
| a05 | 10 | 10 | 10 |
| a23 | 18 | 26 | 36 |
| a24 | 15 | 18 | 25 |
| s01 | 22 | 27 | 27 |
| s02 | 15 | 22 | 30 |
| s03 | 24 | 40 | 44 |
| n04 | 12 | 12 | 14 |
| n06 | 12 | 13 | 14 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

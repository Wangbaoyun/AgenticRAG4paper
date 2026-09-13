# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.9259
- `answer_coverage`: 0.7619
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 51187.1
- `cost_per_question`: 0.077681
- `cost_currency`: CNY
- `mean_latency_s`: 37.618

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 31.4s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 11.1s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 7 | 命中 | 39.6s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 26.4s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 7 | 未命中 | 75.5s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 54.2s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 10 | 命中 | 117.6s |
| n04#0 | How many layers does the original Transf | TRUNCATED | 否 | 1 | 命中 | 33.5s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 24.9s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 18.0s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 11.0s |
| a23#1 | Besides vector databases, what other sou | FAIL | 否 | 8 | 命中 | 123.7s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 24.8s |
| s01#1 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 29.4s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 7 | 命中 | 22.2s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 9 | 未命中 | 57.0s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 26.3s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 25.1s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 3 | 命中 | 11.3s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 8.5s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 27.4s |
| a24#2 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 23.7s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 5 | 未命中 | 51.0s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 27.6s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 65.7s |
| n04#2 | How many layers does the original Transf | TRUNCATED | 是 | 0 | 命中 | 25.4s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 23.2s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.86×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 15 | 33 |
| a05 | 15 | 15 | 15 |
| a23 | 23 | 25 | 32 |
| a24 | 30 | 36 | 41 |
| s01 | 24 | 35 | 63 |
| s02 | 23 | 25 | 52 |
| s03 | 44 | 74 | 80 |
| n04 | 31 | 33 | 44 |
| n06 | 25 | 34 | 45 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

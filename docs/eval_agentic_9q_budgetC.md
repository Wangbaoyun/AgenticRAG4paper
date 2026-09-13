# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.963
- `answer_coverage`: 0.619
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 46595.7
- `cost_per_question`: 0.069141
- `cost_currency`: CNY
- `mean_latency_s`: 33.638

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 6 | 命中 | 45.1s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 5 | 命中 | 42.4s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 30.2s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 26.1s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 6 | 命中 | 46.5s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 41.1s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 9 | 未命中 | 88.2s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 9.0s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 9.6s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 未命中 | 17.4s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 9.3s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 8 | 命中 | 35.2s |
| a24#1 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 30.6s |
| s01#1 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 36.5s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 53.2s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 3 | 未命中 | 53.6s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 12.2s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 14.9s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 3 | 未命中 | 30.7s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 15.4s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 59.0s |
| a24#2 | Which organization is credited with intr | TRUNCATED | 是 | 0 | 未命中 | 23.0s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 4 | 命中 | 51.2s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 40.0s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 9 | 未命中 | 63.3s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 8.9s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 15.6s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.47×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 25 | 31 | 36 |
| a05 | 15 | 15 | 20 |
| a23 | 29 | 32 | 36 |
| a24 | 27 | 30 | 40 |
| s01 | 35 | 35 | 47 |
| s02 | 29 | 34 | 46 |
| s03 | 41 | 44 | 53 |
| n04 | 18 | 19 | 23 |
| n06 | 20 | 26 | 26 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.8889
- `answer_coverage`: 0.7619
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.097995
- `cost_currency`: CNY
- `mean_latency_s`: 36.741

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 25.4s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 7 | 命中 | 22.5s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 53.4s |
| a24#0 | Which organization is credited with intr | REFUSED | 是 | 0 | 未命中 | 17.6s |
| s01#0 | Several papers here propose methods that | TRUNCATED | 否 | 6 | 命中 | 72.9s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 26.9s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 10 | 命中 | 63.7s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 10.8s |
| n06#0 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 27.2s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 11.5s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 9.5s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 51.0s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 18.7s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 153.7s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 30.9s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 4 | 未命中 | 66.1s |
| n04#1 | How many layers does the original Transf | TRUNCATED | 是 | 0 | 命中 | 32.7s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 12.7s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 11.3s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 10.2s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 61.6s |
| a24#2 | Which organization is credited with intr | REFUSED | 是 | 0 | 未命中 | 19.0s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 39.7s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 58.9s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 10 | 命中 | 58.4s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 9.9s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 15.7s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.74×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 15 | 15 |
| a05 | 15 | 15 | 24 |
| a23 | 37 | 43 | 46 |
| a24 | 22 | 22 | 26 |
| s01 | 24 | 53 | 68 |
| s02 | 24 | 25 | 38 |
| s03 | 45 | 60 | 63 |
| n04 | 22 | 22 | 55 |
| n06 | 22 | 22 | 36 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.6667
- `answer_coverage`: 0.5238
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 47183.8
- `cost_per_question`: 0.074716
- `cost_currency`: CNY
- `mean_latency_s`: 30.853

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 5 | 命中 | 32.7s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 7 | 未命中 | 8.4s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 43.1s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 14.2s |
| s01#0 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 33.0s |
| s02#0 | Two papers in this corpus introduce ways | TRUNCATED | 是 | 0 | 未命中 | 43.9s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 63.8s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 14.2s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 14.0s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 15.2s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 4 | 命中 | 8.3s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 6 | 命中 | 37.3s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 14.0s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 37.5s |
| s02#1 | Two papers in this corpus introduce ways | TRUNCATED | 否 | 7 | 命中 | 56.6s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 55.3s |
| n04#1 | How many layers does the original Transf | TRUNCATED | 否 | 1 | 命中 | 32.2s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 15.6s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 10 | 命中 | 21.4s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 7 | 未命中 | 9.0s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 38.7s |
| a24#2 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 16.4s |
| s01#2 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 28.7s |
| s02#2 | Two papers in this corpus introduce ways | TRUNCATED | 是 | 0 | 未命中 | 73.0s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 60.6s |
| n04#2 | How many layers does the original Transf | TRUNCATED | 是 | 0 | 命中 | 26.7s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 19.4s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.31×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 15 | 23 |
| a05 | 15 | 15 | 15 |
| a23 | 24 | 26 | 33 |
| a24 | 22 | 22 | 23 |
| s01 | 26 | 28 | 29 |
| s02 | 36 | 38 | 40 |
| s03 | 38 | 49 | 52 |
| n04 | 23 | 32 | 41 |
| n06 | 23 | 24 | 26 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

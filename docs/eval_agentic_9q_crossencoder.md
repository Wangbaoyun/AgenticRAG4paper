# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.7778
- `answer_coverage`: 0.7143
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.092702
- `cost_currency`: CNY
- `mean_latency_s`: 40.921

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 1 | 命中 | 27.5s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 10.7s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 5 | 命中 | 45.0s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 10 | 命中 | 21.5s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 54.7s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 8 | 命中 | 30.0s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 62.0s |
| n04#0 | How many layers does the original Transf | TRUNCATED | 否 | 7 | 命中 | 34.9s |
| n06#0 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 38.3s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 3 | 命中 | 18.0s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 11.0s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 8 | 命中 | 46.6s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 19.8s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 102.2s |
| s02#1 | Two papers in this corpus introduce ways | TRUNCATED | 是 | 0 | 未命中 | 126.3s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 54.6s |
| n04#1 | How many layers does the original Transf | TRUNCATED | 是 | 0 | 命中 | 25.5s |
| n06#1 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 29.3s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 1 | 命中 | 15.4s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 8.5s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 28.7s |
| a24#2 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 33.9s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 7 | 未命中 | 61.4s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 32.6s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 是 | 0 | 未命中 | 71.6s |
| n04#2 | How many layers does the original Transf | TRUNCATED | 是 | 0 | 命中 | 46.9s |
| n06#2 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 48.1s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.35×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 8 | 10 | 17 |
| a05 | 11 | 12 | 13 |
| a23 | 29 | 30 | 34 |
| a24 | 14 | 17 | 25 |
| s01 | 37 | 40 | 51 |
| s02 | 22 | 25 | 49 |
| s03 | 35 | 45 | 50 |
| n04 | 14 | 16 | 16 |
| n06 | 23 | 26 | 32 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

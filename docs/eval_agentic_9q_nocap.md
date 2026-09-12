# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 1.0
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 59166.6
- `cost_per_question`: 0.109701
- `cost_currency`: CNY
- `mean_latency_s`: 41.757

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 5 | 命中 | 34.9s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 9.6s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 60.3s |
| a24#0 | Which organization is credited with intr | TRUNCATED | 否 | 3 | 命中 | 28.7s |
| s01#0 | Several papers here propose methods that | TRUNCATED | 否 | 6 | 未命中 | 42.7s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 4 | 命中 | 31.3s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 9 | 未命中 | 66.2s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.2s |
| n06#0 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 53.8s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 32.8s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 12.6s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 45.4s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 21.0s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 否 | 10 | 命中 | 74.8s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 29.2s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 9 | 命中 | 136.4s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 15.0s |
| n06#1 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 32.7s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 59.1s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 10.3s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 56.3s |
| a24#2 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 20.4s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 4 | 命中 | 57.8s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 44.9s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 10 | 命中 | 109.4s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 13.0s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 17.5s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.55×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 25 | 36 | 82 |
| a05 | 15 | 15 | 15 |
| a23 | 34 | 43 | 44 |
| a24 | 26 | 32 | 41 |
| s01 | 31 | 37 | 65 |
| s02 | 24 | 25 | 25 |
| s03 | 60 | 65 | 74 |
| n04 | 22 | 23 | 24 |
| n06 | 23 | 46 | 67 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

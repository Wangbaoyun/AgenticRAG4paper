# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.963
- `answer_coverage`: 0.8095
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 46246.9
- `cost_per_question`: 0.068497
- `cost_currency`: CNY
- `mean_latency_s`: 30.828

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 29.6s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 5 | 未命中 | 27.7s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 45.8s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 11.9s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 37.7s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 8 | 命中 | 35.6s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 10 | 未命中 | 71.9s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 15.6s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 11.9s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 5 | 命中 | 25.6s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 11.4s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 8 | 命中 | 46.4s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 22.1s |
| s01#1 | Several papers here propose methods that | SUCCESS | 否 | 4 | 命中 | 37.8s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 20.5s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 9 | 命中 | 111.7s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.7s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 9.3s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 20.7s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 10.3s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 24.3s |
| a24#2 | Which organization is credited with intr | TRUNCATED | 是 | 0 | 未命中 | 18.6s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 51.9s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 4 | 命中 | 27.9s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 9 | 未命中 | 53.7s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.3s |
| n06#2 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 29.2s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.83×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 24 | 34 |
| a05 | 15 | 15 | 41 |
| a23 | 30 | 32 | 36 |
| a24 | 24 | 34 | 34 |
| s01 | 31 | 34 | 44 |
| s02 | 23 | 25 | 31 |
| s03 | 46 | 47 | 61 |
| n04 | 22 | 23 | 24 |
| n06 | 22 | 22 | 45 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

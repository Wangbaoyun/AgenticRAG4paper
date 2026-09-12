# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 1.0
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.031694
- `cost_currency`: CNY
- `mean_latency_s`: 12.124

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 18.2s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 9.1s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 5 | 命中 | 10.6s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 1 | 命中 | 6.8s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 3 | 未命中 | 17.6s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 2 | 命中 | 26.0s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 4 | 命中 | 21.9s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 5.4s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 4.3s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 5.5s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 7.0s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 2 | 命中 | 9.1s |
| a24#1 | Which organization is credited with intr | SUCCESS | 否 | 1 | 命中 | 7.4s |
| s01#1 | Several papers here propose methods that | SUCCESS | 否 | 3 | 未命中 | 22.1s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 2 | 命中 | 22.7s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 4 | 命中 | 24.3s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 4.4s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 5.2s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 5.0s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 5.1s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 3 | 命中 | 9.8s |
| a24#2 | Which organization is credited with intr | SUCCESS | 否 | 1 | 命中 | 8.9s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 4 | 命中 | 14.0s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 2 | 命中 | 16.1s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | SUCCESS | 否 | 2 | 未命中 | 30.8s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 3.4s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 6.4s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.24×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 11 | 11 | 11 |
| a05 | 11 | 11 | 11 |
| a23 | 11 | 11 | 11 |
| a24 | 11 | 11 | 11 |
| s01 | 11 | 11 | 11 |
| s02 | 11 | 11 | 11 |
| s03 | 11 | 11 | 11 |
| n04 | 10 | 10 | 10 |
| n06 | 10 | 10 | 10 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

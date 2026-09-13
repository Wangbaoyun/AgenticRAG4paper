# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 1.0
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `tokens_per_question`: 51896.6
- `cost_per_question`: 0.095936
- `cost_currency`: CNY
- `mean_latency_s`: 41.905

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 27.6s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 12.2s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 57.0s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 3 | 命中 | 28.3s |
| s01#0 | Several papers here propose methods that | SUCCESS | 否 | 5 | 命中 | 43.2s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 7 | 命中 | 31.5s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 3 | 未命中 | 110.1s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 16.9s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 10.5s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 2 | 命中 | 16.1s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 未命中 | 9.3s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 10 | 命中 | 61.6s |
| a24#1 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 22.2s |
| s01#1 | Several papers here propose methods that | TRUNCATED | 否 | 7 | 未命中 | 123.8s |
| s02#1 | Two papers in this corpus introduce ways | SUCCESS | 否 | 8 | 命中 | 61.8s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 10 | 命中 | 108.6s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 11.8s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 12.6s |
| a01#2 | How many parameters does GPT-3 have? | SUCCESS | 否 | 7 | 命中 | 23.3s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 2 | 命中 | 25.8s |
| a23#2 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 52.2s |
| a24#2 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 27.5s |
| s01#2 | Several papers here propose methods that | SUCCESS | 否 | 6 | 命中 | 70.1s |
| s02#2 | Two papers in this corpus introduce ways | SUCCESS | 否 | 5 | 命中 | 29.1s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 9 | 命中 | 101.8s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 12.0s |
| n06#2 | How many parameters does GPT-4 have? | TRUNCATED | 是 | 0 | 命中 | 24.6s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`1.76×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 15 | 15 | 25 |
| a05 | 15 | 15 | 15 |
| a23 | 33 | 35 | 37 |
| a24 | 26 | 27 | 31 |
| s01 | 34 | 37 | 57 |
| s02 | 23 | 24 | 40 |
| s03 | 55 | 62 | 72 |
| n04 | 18 | 19 | 27 |
| n06 | 15 | 18 | 32 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.5926
- `answer_coverage`: 0.4286
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `cost_per_question`: 0.812248
- `cost_currency`: USD
- `mean_latency_s`: 15.638

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| a01#0 | How many parameters does GPT-3 have? | SUCCESS | 否 | 1 | 命中 | 28.1s |
| a05#0 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 3 | 命中 | 11.2s |
| a23#0 | Besides vector databases, what other sou | SUCCESS | 否 | 8 | 命中 | 66.7s |
| a24#0 | Which organization is credited with intr | SUCCESS | 否 | 2 | 命中 | 23.7s |
| s01#0 | Several papers here propose methods that | TRUNCATED | 是 | 0 | 未命中 | 39.7s |
| s02#0 | Two papers in this corpus introduce ways | SUCCESS | 否 | 6 | 命中 | 27.5s |
| s03#0 | Compare how MemGPT and GraphRAG each ext | TRUNCATED | 否 | 10 | 命中 | 110.7s |
| n04#0 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 12.8s |
| n06#0 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 19.1s |
| a01#1 | How many parameters does GPT-3 have? | SUCCESS | 否 | 4 | 命中 | 14.3s |
| a05#1 | How many tokens was DeepSeek-V3 pre-trai | SUCCESS | 否 | 5 | 未命中 | 16.2s |
| a23#1 | Besides vector databases, what other sou | SUCCESS | 否 | 9 | 命中 | 31.0s |
| a24#1 | Which organization is credited with intr | TRUNCATED | 否 | 2 | 命中 | 21.2s |
| s01#1 | Several papers here propose methods that | REFUSED | 是 | 0 | 未命中 | 0.0s |
| s02#1 | Two papers in this corpus introduce ways | REFUSED | 是 | 0 | 未命中 | 0.0s |
| s03#1 | Compare how MemGPT and GraphRAG each ext | REFUSED | 是 | 0 | 未命中 | 0.0s |
| n04#1 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 0.0s |
| n06#1 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 0.0s |
| a01#2 | How many parameters does GPT-3 have? | REFUSED | 是 | 0 | 未命中 | 0.0s |
| a05#2 | How many tokens was DeepSeek-V3 pre-trai | REFUSED | 是 | 0 | 未命中 | 0.0s |
| a23#2 | Besides vector databases, what other sou | REFUSED | 是 | 0 | 未命中 | 0.0s |
| a24#2 | Which organization is credited with intr | REFUSED | 是 | 0 | 未命中 | 0.0s |
| s01#2 | Several papers here propose methods that | REFUSED | 是 | 0 | 未命中 | 0.0s |
| s02#2 | Two papers in this corpus introduce ways | REFUSED | 是 | 0 | 未命中 | 0.0s |
| s03#2 | Compare how MemGPT and GraphRAG each ext | REFUSED | 是 | 0 | 未命中 | 0.0s |
| n04#2 | How many layers does the original Transf | REFUSED | 是 | 0 | 命中 | 0.0s |
| n06#2 | How many parameters does GPT-4 have? | REFUSED | 是 | 0 | 命中 | 0.0s |

## 运行间稳定性（**先看这张表再看汇总**）

- 每题重复次数：`3`
- 逐题成本极差中位数（最贵/最便宜）：`2.69×`

| 题号 | LLM 调用 最少 | 中位 | 最多 |
| --- | --- | --- | --- |
| a01 | 24 | 296 | 367 |
| a05 | 39 | 321 | 367 |
| a23 | 70 | 344 | 367 |
| a24 | 95 | 367 | 367 |
| s01 | 137 | 367 | 367 |
| s02 | 160 | 367 | 367 |
| s03 | 235 | 367 | 367 |
| n04 | 259 | 367 | 367 |
| n06 | 281 | 367 | 367 |

> 极差远大于 1 时，题目之间的对比只在**同一批重复**内成立；拿不同配置的单次运行互比会得到符号都可能相反的结论。

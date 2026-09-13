# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.963
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.037
- `tokens_per_question`: 43114.1
- `cost_per_question`: 0.063765
- `cost_currency`: CNY
- `mean_latency_s`: 32.675

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | What average accuracy did Chinchilla rea | SUCCESS | 否 | 4 | 命中 | 39.7s |
| 2 | In the Game of 24 task, what success rat | SUCCESS | 否 | 3 | 命中 | 12.3s |
| 3 | What pass@1 accuracy did Reflexion achie | SUCCESS | 否 | 4 | 命中 | 23.4s |
| 4 | Which larger model does Mistral 7B outpe | SUCCESS | 否 | 2 | 命中 | 18.5s |
| 5 | What two kinds of memory does the origin | SUCCESS | 否 | 3 | 命中 | 30.8s |
| 6 | What mechanism does Self-RAG use to deci | SUCCESS | 否 | 4 | 命中 | 20.9s |
| 7 | How does Corrective RAG deal with retrie | SUCCESS | 否 | 5 | 命中 | 21.9s |
| 8 | Why does GraphRAG build an entity graph  | SUCCESS | 否 | 7 | 命中 | 52.9s |
| 9 | What two things does the ReAct approach  | SUCCESS | 否 | 3 | 命中 | 35.4s |
| 10 | How does Toolformer learn to call extern | SUCCESS | 否 | 7 | 命中 | 49.6s |
| 11 | How does MemGPT manage its context, and  | SUCCESS | 否 | 8 | 命中 | 56.5s |
| 12 | What does Gorilla connect a language mod | SUCCESS | 否 | 4 | 命中 | 23.6s |
| 13 | What attention mechanism does DeepSeek-V | SUCCESS | 否 | 2 | 命中 | 17.9s |
| 14 | What existing organizational concept doe | SUCCESS | 否 | 5 | 命中 | 23.4s |
| 15 | Which mathematics benchmark does AutoGen | SUCCESS | 否 | 3 | 命中 | 32.9s |
| 16 | Which reasoning task categories does cha | SUCCESS | 否 | 6 | 命中 | 60.3s |
| 17 | What does InstructGPT train language mod | SUCCESS | 否 | 5 | 未命中 | 31.2s |
| 18 | What is the core iterative mechanism of  | SUCCESS | 否 | 4 | 未命中 | 35.5s |
| 19 | For a given compute budget, what does LL | TRUNCATED | 否 | 3 | 未命中 | 92.3s |
| 20 | What ability is central to the RAP retri | SUCCESS | 否 | 4 | 命中 | 25.6s |
| 21 | What applications does the SWE-agent des | SUCCESS | 否 | 6 | 命中 | 69.4s |
| 22 | What is the superconducting transition t | REFUSED | 是 | 0 | 命中 | 9.4s |
| 23 | Who won the 2018 FIFA World Cup? | REFUSED | 是 | 0 | 命中 | 7.8s |
| 24 | What is the melting point of gallium at  | REFUSED | 是 | 0 | 命中 | 9.1s |
| 25 | What BLEU score did the original Transfo | REFUSED | 是 | 0 | 命中 | 9.6s |
| 26 | What learning rate schedule and model si | TRUNCATED | 否 | 1 | 命中 | 28.2s |
| 27 | What is the population of Iceland? | REFUSED | 是 | 0 | 命中 | 44.5s |

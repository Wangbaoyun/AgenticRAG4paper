# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 1.0
- `answer_coverage`: 0.9524
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `answer_language_mix`: {'zh': 10, 'en': 11, 'none': 6}
- `tokens_per_question`: 40160.7
- `cost_per_question`: 0.057129
- `cost_currency`: CNY
- `mean_latency_s`: 27.673

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | What average accuracy did Chinchilla rea | TRUNCATED | 否 | 3 | 命中 | 40.6s |
| 2 | In the Game of 24 task, what success rat | SUCCESS | 否 | 3 | 命中 | 18.9s |
| 3 | What pass@1 accuracy did Reflexion achie | TRUNCATED | 否 | 1 | 命中 | 11.0s |
| 4 | Which larger model does Mistral 7B outpe | SUCCESS | 否 | 2 | 命中 | 16.8s |
| 5 | What two kinds of memory does the origin | SUCCESS | 否 | 4 | 命中 | 19.5s |
| 6 | What mechanism does Self-RAG use to deci | SUCCESS | 否 | 6 | 命中 | 40.9s |
| 7 | How does Corrective RAG deal with retrie | SUCCESS | 否 | 6 | 命中 | 50.4s |
| 8 | Why does GraphRAG build an entity graph  | SUCCESS | 否 | 8 | 命中 | 54.2s |
| 9 | What two things does the ReAct approach  | SUCCESS | 否 | 9 | 命中 | 19.1s |
| 10 | How does Toolformer learn to call extern | SUCCESS | 否 | 6 | 命中 | 49.2s |
| 11 | How does MemGPT manage its context, and  | SUCCESS | 否 | 7 | 命中 | 28.7s |
| 12 | What does Gorilla connect a language mod | SUCCESS | 否 | 5 | 命中 | 15.2s |
| 13 | What attention mechanism does DeepSeek-V | SUCCESS | 否 | 5 | 命中 | 46.3s |
| 14 | What existing organizational concept doe | SUCCESS | 否 | 3 | 命中 | 35.7s |
| 15 | Which mathematics benchmark does AutoGen | SUCCESS | 否 | 2 | 命中 | 26.2s |
| 16 | Which reasoning task categories does cha | SUCCESS | 否 | 7 | 命中 | 38.3s |
| 17 | What does InstructGPT train language mod | SUCCESS | 否 | 6 | 命中 | 35.9s |
| 18 | What is the core iterative mechanism of  | SUCCESS | 否 | 7 | 命中 | 18.5s |
| 19 | For a given compute budget, what does LL | SUCCESS | 否 | 3 | 命中 | 47.6s |
| 20 | What ability is central to the RAP retri | SUCCESS | 否 | 4 | 未命中 | 34.6s |
| 21 | What applications does the SWE-agent des | SUCCESS | 否 | 5 | 命中 | 42.7s |
| 22 | What is the superconducting transition t | REFUSED | 是 | 0 | 命中 | 9.5s |
| 23 | Who won the 2018 FIFA World Cup? | REFUSED | 是 | 0 | 命中 | 8.8s |
| 24 | What is the melting point of gallium at  | REFUSED | 是 | 0 | 命中 | 9.5s |
| 25 | What BLEU score did the original Transfo | REFUSED | 是 | 0 | 命中 | 11.1s |
| 26 | What learning rate schedule and model si | REFUSED | 是 | 0 | 命中 | 12.1s |
| 27 | What is the population of Iceland? | REFUSED | 是 | 0 | 命中 | 6.2s |

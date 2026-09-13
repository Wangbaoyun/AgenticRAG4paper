# scitrace 评测报告

## 汇总

- `total`: 27
- `behavior_correctness`: 0.963
- `answer_coverage`: 0.8571
- `citation_resolution_rate`: 1.0
- `hallucination_rate`: 0.0
- `answer_language_mix`: {'zh': 10, 'mixed': 2, 'en': 10, 'none': 5}
- `tokens_per_question`: 54417.8
- `cost_per_question`: 0.10158
- `cost_currency`: CNY
- `mean_latency_s`: 40.562

## 逐题结果

| # | 问题 | 状态 | 拒答 | 引用 | 关键词 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | What average accuracy did Chinchilla rea | SUCCESS | 否 | 2 | 未命中 | 35.6s |
| 2 | In the Game of 24 task, what success rat | SUCCESS | 否 | 4 | 命中 | 27.8s |
| 3 | What pass@1 accuracy did Reflexion achie | SUCCESS | 否 | 2 | 命中 | 36.7s |
| 4 | Which larger model does Mistral 7B outpe | SUCCESS | 否 | 2 | 命中 | 59.8s |
| 5 | What two kinds of memory does the origin | SUCCESS | 否 | 6 | 命中 | 30.2s |
| 6 | What mechanism does Self-RAG use to deci | SUCCESS | 否 | 8 | 命中 | 42.9s |
| 7 | How does Corrective RAG deal with retrie | TRUNCATED | 否 | 7 | 命中 | 92.3s |
| 8 | Why does GraphRAG build an entity graph  | SUCCESS | 否 | 6 | 命中 | 57.0s |
| 9 | What two things does the ReAct approach  | SUCCESS | 否 | 4 | 命中 | 24.4s |
| 10 | How does Toolformer learn to call extern | SUCCESS | 否 | 7 | 命中 | 98.6s |
| 11 | How does MemGPT manage its context, and  | SUCCESS | 否 | 6 | 命中 | 34.7s |
| 12 | What does Gorilla connect a language mod | SUCCESS | 否 | 6 | 命中 | 26.3s |
| 13 | What attention mechanism does DeepSeek-V | SUCCESS | 否 | 4 | 命中 | 30.6s |
| 14 | What existing organizational concept doe | SUCCESS | 否 | 3 | 命中 | 10.7s |
| 15 | Which mathematics benchmark does AutoGen | SUCCESS | 否 | 2 | 命中 | 27.6s |
| 16 | Which reasoning task categories does cha | TRUNCATED | 否 | 8 | 命中 | 70.6s |
| 17 | What does InstructGPT train language mod | SUCCESS | 否 | 5 | 命中 | 35.5s |
| 18 | What is the core iterative mechanism of  | SUCCESS | 否 | 7 | 命中 | 35.3s |
| 19 | For a given compute budget, what does LL | TRUNCATED | 否 | 4 | 命中 | 113.7s |
| 20 | What ability is central to the RAP retri | SUCCESS | 否 | 3 | 未命中 | 20.1s |
| 21 | What applications does the SWE-agent des | SUCCESS | 否 | 8 | 未命中 | 58.0s |
| 22 | What is the superconducting transition t | REFUSED | 是 | 0 | 命中 | 11.7s |
| 23 | Who won the 2018 FIFA World Cup? | REFUSED | 是 | 0 | 命中 | 7.1s |
| 24 | What is the melting point of gallium at  | REFUSED | 是 | 0 | 命中 | 9.0s |
| 25 | What BLEU score did the original Transfo | REFUSED | 是 | 0 | 命中 | 14.5s |
| 26 | What learning rate schedule and model si | TRUNCATED | 否 | 1 | 命中 | 36.6s |
| 27 | What is the population of Iceland? | REFUSED | 是 | 0 | 命中 | 47.8s |

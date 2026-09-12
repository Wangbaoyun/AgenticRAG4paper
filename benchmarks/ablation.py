"""检索策略消融：24 个易混淆查询 × 4 种策略。

用法：`PYTHONPATH=src python benchmarks/ablation.py`

**查询集的设计比脚本本身重要**：8 个主题迥异的查询会给出与事实相反的结论
（见 docs/EXPERIMENTS.md 实验一）。这里刻意让同主题论文互为干扰项。
"""

import sys, asyncio, logging, time, json
sys.path.insert(0, "src"); logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from scitrace.config import load_settings, RetrievalSettings
from scitrace.factory import build_services
from scitrace.adapters.llm.reranker import CrossEncoderReranker
from scitrace.pipeline.retrieval import Retriever

# 刻意构造**易混淆**的查询：每组的候选论文主题高度重叠，
# 检索器必须靠细节区分，而不是靠"整篇论文主题不同"这种廉价信号。
CASES = [
    # —— 检索增强四兄弟：RAG / Self-RAG / CRAG / GraphRAG 互相干扰 ——
    ("What memory types does the original retrieval-augmented generation model combine?", "01_Lewis2020"),
    ("Which system uses reflection tokens to decide when to retrieve?",                  "02_Asai2023"),
    ("Which approach uses a lightweight retrieval evaluator with a confidence degree?",  "03_Yan2024"),
    ("Which method builds an entity graph and summarizes communities?",                  "04_Edge2024"),
    ("Which paper scales agentic retrieval-augmented generation hierarchically?",        "06_Du2025"),
    ("Which survey organizes agentic RAG systems into a taxonomy?",                      "05_Singh2025"),
    # —— Agent 四兄弟：ReAct / Reflexion / Toolformer / Gorilla ——
    ("Which method interleaves reasoning traces with task-specific actions?",            "01_ReAct"),
    ("Which method uses verbal self-reflection stored in an episodic memory buffer?",     "05_Reflexion"),
    ("Which method teaches itself to call APIs by self-supervision?",                     "03_Toolformer"),
    ("Which method connects a language model to massive numbers of APIs?",                "06_Gorilla"),
    ("Which method treats an LLM as an operating system with paged memory?",              "07_MemGPT"),
    ("Which method refines its own output iteratively using self-feedback?",              "09_Self-Refine"),
    # —— 推理三兄弟：CoT / ToT / RAP ——
    ("Which method elicits reasoning by generating intermediate natural language steps?", "02_Chain-of-Thought"),
    ("Which method searches over multiple reasoning paths with lookahead and backtracking?", "04_Tree_of_Thoughts"),
    ("Which method combines retrieval with planning for multimodal agents?",              "08_RAP"),
    # —— 基座模型：LLaMA / Llama-2 / Mistral / GPT-3 / BERT ——
    ("Which model is a 7B model trained on publicly available data with grouped-query attention?", "13_Mistral-7B"),
    ("Which model used only publicly available datasets and introduced RMSNorm?",         "06_LLaMA-1"),
    ("Which model was fine-tuned with RLHF on a mix of helpfulness and safety data?",     "07_Llama-2"),
    ("Which model demonstrated few-shot learning at scale without gradient updates?",     "03_GPT-3"),
    ("Which model introduced masked language modeling with bidirectional pretraining?",   "BERT_Pre-training"),
    ("Which model used a Mixture-of-Experts architecture with multi-head latent attention?", "09_DeepSeek-V2"),
    # —— 训练方法：Chinchilla / InstructGPT ——
    ("Which paper derived compute-optimal training ratios between model size and data?",  "04_Chinchilla"),
    ("Which paper aligned language models to instructions using human feedback?",         "05_InstructGPT"),
    ("Which paper analyzes the training compute of frontier models with a new scaling law?", "10_DeepSeek-V3"),
]
STRATEGIES = ["dense", "dense_mmr", "hybrid_rrf", "hybrid_rrf_rerank"]

async def main():
    s = load_settings()
    svc = build_services(s, load_index=True)
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    print(f"语料 {len(svc.vector_index)} 片段 / {len(svc.sources)} 文献；查询 {len(CASES)} 个（易混淆设计）\n")

    async def run(strategy):
        r = Retriever(vector_index=svc.vector_index, fulltext_index=svc.fulltext_index,
                      embedder=svc.embedder,
                      settings=RetrievalSettings(strategy=strategy, k=10),
                      reranker=reranker if strategy.endswith("rerank") else None)
        h1 = h5 = h10 = 0; rrs = []; durs = []
        for q, target in CASES:
            t0 = time.perf_counter(); res = await r.retrieve(q, k=10)
            durs.append(time.perf_counter()-t0)
            rank = next((i for i, h in enumerate(res, 1)
                         if target in svc.sources[h.fragment.source_key].rel_path), None)
            if rank:
                h10 += 1; h5 += rank <= 5; h1 += rank == 1; rrs.append(1.0/rank)
            else: rrs.append(0.0)
        return h1, h5, h10, sum(rrs)/len(rrs), sum(durs)/len(durs)

    await run("dense")   # 预热
    print(f"{'策略':<20}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}{'耗时':>10}")
    out = {}
    for st in STRATEGIES:
        a,b,c,m,d = await run(st)
        out[st] = {"hit@1":a,"hit@5":b,"hit@10":c,"mrr":round(m,3),"latency_s":round(d,3)}
        print(f"{st:<20}{a:>5}/24{b:>5}/24{c:>6}/24{m:>8.3f}{d:>9.3f}s", flush=True)
    Path("/tmp/ablation").mkdir(exist_ok=True)
    json.dump({"n": len(CASES), "results": out}, open("/tmp/ablation/hard.json","w"), ensure_ascii=False, indent=2)
    await svc.aclose()

asyncio.run(main())

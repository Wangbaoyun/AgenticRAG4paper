# scitrace

> **证据可溯源**的科研文献问答系统 —— 检索 → 取证 → 合成，每一句结论都能回到原文片段。

给定一批 PDF（论文、预印本、技术报告），`scitrace` 回答你的问题，并给出**可点击回溯到原文
片段**的引用与参考文献列表；当证据不足时，它会明确拒答，而不是编造。

```bash
stc index ./my_papers              # 建索引：解析 → 分块 → 嵌入 → 双索引
stc ask "这篇论文的核心贡献是什么？"   # 问答：检索 → 取证 → 合成带引用的答案
stc search "retrieval augmented generation"   # 只检索，看证据长什么样
```

---

## ⚠️ 与 PaperQA2 的关系（请先读这一段）

`scitrace` 是**独立实现**的项目。它不是 PaperQA2 的 fork，**不包含**其任何源代码、Prompt
字符串、注释或文档文本。

它的**设计范式**受 PaperQA2 的公开工作启发：

- 论文：Skarlinski et al., *Language agents achieve superhuman synthesis of scientific
  knowledge*, [arXiv:2409.13740](https://arxiv.org/abs/2409.13740)
- 代码：[Future-House/paper-qa](https://github.com/Future-House/paper-qa)（Apache-2.0，
  Copyright 2024 FutureHouse）

具体而言，**"检索论文 → 收集证据（LLM 相关性打分 + 上下文摘要）→ 合成带引用答案"的三阶段
智能体流程**，以及"元数据一等公民""双索引互补""配置指纹决定索引名"等工程思想，来自上述
公开工作。这些属于**方法/思想**层面，`scitrace` 在其基础上独立设计了自己的代码结构、
数据模型、Prompt 文本、引用方案与检索策略。

`scitrace` 与 FutureHouse 无从属关系，亦未获其背书。"PaperQA" 是 FutureHouse 的标识。

**本项目不含任何来自上游的受著作权保护的表达。** 这一主张由两道机械化证据支持：

- `tools/similarity_audit.py` —— 与上游 git HEAD 逐版本比对的自动化审计（n-gram 重叠、
  逐行复制、字符串字面量相似度、依赖、资产哈希）；最新报告见 `docs/AUDIT.md`。
- `docs/PROVENANCE.md` —— 记录开发流程、隔离纪律与污染事件，说明"思想层面的独立"如何被保证。

> 说明：本审计能证明的是"**无文本复制**"；"思想独立"依赖流程纪律与开发记录。
> 我们选择把两件事分开陈述，而不是含糊地宣称"100% 原创"。

---

## 核心特性

| 特性 | 说明 |
| --- | --- |
| **证据可溯源** | 答案中的每个引用键都指向具体片段；证据不足时明确拒答 |
| **自研 Agent 运行时** | 工具协议、循环、状态机、超时兜底、预算闸门全部自研，不依赖第三方 Agent 框架 |
| **可插拔检索内核** | 向量+MMR / BM25+向量 RRF 融合 / Cross-Encoder 重排，可配置、可对照、可消融 |
| **中文文献优先** | 中文分块与句子边界、中文引用格式、国产模型（DeepSeek 等）优先、本地 embedding 默认 |
| **成本与延迟治理** | 证据缓存、token 预算闸门、工具并行、模型分级路由 |
| **元数据一等公民** | 入库时并发补全 Crossref / Semantic Scholar / OpenAlex，单源失败自动降级 |
| **双索引互补** | 向量索引（语义召回）+ 全文索引（关键词精确召回，增量更新） |

---

## 架构总览

```
                 ┌─────────────── config（配置指纹 → 索引名）───────────────┐
                 ▼                                                        │
  PDF ──► parsers ──► chunking ──► embedding ──► ┌ vector index ┐         │
                                                 └ fulltext index┘         │
                 │                                                        │
                 ▼                                                        │
  question ──► retriever ──► screener ──► evidence ──► synthesizer ──► answer + citations
              （可插拔）      （可插拔）                  │
                 ▲                                      ▼
                 └──────────── agent runtime（自研循环 + 工具 + 预算）────┘
```

模块边界详见 `docs/SPEC.md`；目录结构：

```
src/scitrace/
  domain/     纯领域模型（零外部依赖）
  ports/      接口协议（typing.Protocol）
  adapters/   parsers / indexes / metadata / llm 的具体实现
  pipeline/   ingest · chunking · retrieval · screening · synthesis
  agent/      自研运行时：loop · tools · state · budget
  config/     分层配置与索引指纹
  service/    持久化、会话历史、缓存
  util/       通用工具
  prompts/    提示词（独立可版本化）
  cli.py      命令行入口
```

---

## 快速开始

```bash
# 1) 环境（Python 3.11+）
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pypdf,llm,local-embedding]"

# 2) 配置（~/.scitrace/settings/default.json，或环境变量 SCITRACE_*）
cp .env.example .env    # 填入模型 API key

# 3) 用起来
stc config show
stc index ./my_papers
stc ask "这批论文在方法上有什么共同点？"
```

---

## 开发

```bash
pytest                                   # 单测 + 行为契约测试
python tools/similarity_audit.py \
    --target . --reference ../PaperQA \
    --report docs/AUDIT.md               # 反抄袭审计（CI 必跑）
```

---

## 实现状态

M0–M6 最小可行内核已完成，各项原创增量的落点如下：

| 增量 | 状态 | 落点 |
| --- | --- | --- |
| ① 自研 Agent 运行时 | 已实现 | `src/scitrace/agent/`（工具协议、循环、状态机、超时兜底、参数容错） |
| ② 可插拔检索内核 + 消融 | 已实现 | `pipeline/retrieval.py` 四路策略；`pipeline/screening.py` 两种筛选实现 |
| ③ 中文文献适配 | 已实现 | `util/tokenize_zh.py`（bigram）、`pipeline/chunking.py`（中文句边界）、提示词中英双语 |
| ④ 成本与延迟治理 | 已实现 | `agent/budget.py`（闸门）、`Services.usage`（累计口径）、模型分级路由 |
| ⑤ 自主评测体系 | 骨架就绪 | `benchmarks/qa_eval.py`（指标与 SPEC §1.3 质量目标一一对应） |

**洁净室证据链**（详见 `docs/PROVENANCE.md`）：

- `docs/SPEC.md` —— 冻结的行为规格，含三轮修订记录；
- `tools/similarity_audit.py` + `docs/AUDIT.md` —— 与上游 git HEAD 逐版本比对的审计，
  5 项检查全部通过，**最长公共 token 连续片段为 0**；
- 审计工具自身经过**植入式对抗验证**：从上游植入真实代码后必须失败，
  否则说明它已退化为恒通过。

**验证覆盖**：单元测试 + 真实适配器集成测试（真实 pypdf / tantivy / numpy +
内存替换嵌入与 LLM）+ SPEC §8 的 C1–C8 行为契约。

> 说明：真实的 LLM 与本地嵌入模型端到端运行需要 API key 与模型下载，
> 未在本仓库内执行；端到端链路由上述集成测试覆盖。

---

## 许可

Apache License 2.0，Copyright 2026 wby。见 `LICENSE` 与 `NOTICE`。

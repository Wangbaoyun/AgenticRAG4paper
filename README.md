# scitrace

> **证据可溯源**的科研文献问答系统 —— 检索 → 取证 → 合成，每一句结论都能回到原文片段。

给定一批 PDF（论文、预印本、技术报告），`scitrace` 回答你的问题，并给出**可回溯到原文
片段**的引用与参考文献列表；当证据不足时，它会明确拒答，而不是编造。

```bash
stc index ./my_papers                        # 建索引：解析 → 分块 → 嵌入 → 双索引
stc ask "这篇论文的核心贡献是什么？"            # 问答：检索 → 取证 → 合成带引用的答案
stc ask --mode agentic "两篇论文的方法有何不同？"  # 由模型自己决定检索与取证顺序
stc search "retrieval augmented generation"  # 只检索，看证据长什么样
stc status                                   # 索引状态
stc sessions                                 # 历史问答
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
- `docs/PROVENANCE.md` —— 记录开发流程、隔离纪律与**污染事件**，说明"思想层面的独立"
  如何被保证。

> 说明：本审计能证明的是"**无文本复制**"；"思想独立"依赖流程纪律与开发记录。
> 我们选择把两件事分开陈述，而不是含糊地宣称"100% 原创"。

**一处必须说明的边界**：`docs/PROVENANCE.md` 的污染事件 **E3** 记录了在实现完成之后、
为回答"PaperQA2 的 agentic 模式怎么做"而阅读了上游 agentic 相关源码。若据此修改 agentic
实现，`scitrace` 在**该区域**的定位应从"洁净室重写"收窄为"读后重实现"。
当前 agentic 的两项改进（摘要篇幅上限、已筛片段去重）都能由本项目自己的测量独立推出，
来源记为"由本项目测量得出"，但这条边界应随代码一起被读者知晓。

---

## 目录

- [项目结构](#项目结构)
- [数据流](#数据流)
- [从零开始的阅读与学习路径](#从零开始的阅读与学习路径)
- [快速开始](#快速开始)
- [验证与质量保障](#验证与质量保障)
- [实验体系与如何读它](#实验体系与如何读它)
- [实现状态与已知边界](#实现状态与已知边界)
- [开发约定](#开发约定)
- [许可](#许可)

---

## 项目结构

### 顶层

```
AgenticRAG4paper/
├── src/scitrace/        # 全部实现（12,408 行）
├── tests/               # 28 个测试文件，1,150 条测试（10,662 行）
├── docs/                # 规格、实验、审计、来源记录
├── benchmarks/          # 评测骨架与消融脚本
├── tools/               # similarity_audit.py（反抄袭审计）
├── my_papers/           # 语料（已 gitignore，需自备）
├── pyproject.toml
└── README.md
```

### 分层架构（核心）

依赖方向**单向向下**，且由 `tests/test_layering.py` 用 AST 静态检查强制执行：

```
                    ┌──────────────────────────────────────┐
                    │  cli.py / api.py      入口            │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  factory.py      唯一的装配点         │
                    │  （只有这里知道"用哪个实现"）           │
                    └───────────────┬──────────────────────┘
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
      │  pipeline/    │   │   agent/      │   │   service/    │
      │  ingest       │   │   runtime     │   │   持久化       │
      │  chunking     │   │   tools       │   │   会话历史     │
      │  retrieval    │   │   state       │   └───────────────┘
      │  screening    │   │   budget      │
      │  synthesis    │   └───────┬───────┘
      └───────┬───────┘           │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │      ports/       │  接口协议（Protocol）
              │  定义"能做什么"     │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │      domain/      │  纯数据模型（零外部依赖）
              └───────────────────┘
                        ▲
              ┌─────────┴─────────┐
              │    adapters/      │  具体实现：parsers / indexes
              │  定义"怎么做"       │           / metadata / llm
              └───────────────────┘
```

**关键规则**：`pipeline/` 只依赖 `domain` / `ports` / `util` / `config`，
**不依赖任何适配器实现**。这条规则由机械检查守着，因为破坏它不会让任何测试失败——
编译能过、测试能过，直到有人真想换一个 PDF 解析后端时才发现业务逻辑里到处是它的引用。

### 各包职责与规模

| 包 | 文件 | 行数 | 职责 | 关键文件 |
| --- | --- | --- | --- | --- |
| `domain/` | 6 | 975 | 纯领域模型：`Document` / `Fragment` / `Evidence` / `Answer` / `Session` / `Usage`。**零外部依赖**，是整个系统的词汇表 | `session.py`（`Usage` / `SessionStatus`）、`evidence.py` |
| `ports/` | 9 | 752 | 接口协议（`typing.Protocol`）：`LLMClient` / `Embedder` / `VectorIndex` / `FulltextIndex` / `Parser` / `Screener` / `Synthesizer` / `Retriever` | `llm.py`、`index.py` |
| `adapters/` | 20 | 3,984 | 具体实现：`parsers/`（pypdf / plaintext）、`indexes/`（numpy 向量 / tantivy 全文）、`metadata/`（Crossref / Semantic Scholar / OpenAlex）、`llm/`（litellm 客户端 + 重试） | `llm/litellm_client.py` |
| `pipeline/` | 7 | 2,593 | 五个阶段：摄入、分块、检索、筛选、合成。**业务逻辑所在，与实现解耦** | `retrieval.py`（四路策略）、`screening.py`、`synthesis.py` |
| `agent/` | 5 | 1,103 | 自研 Agent 运行时：工具协议、循环、状态机、预算闸门、历史压缩 | `runtime.py`、`tools.py`、`state.py` |
| `config/` | 2 | 702 | 分层配置（默认值 → 配置文件 → 环境变量）、索引指纹 | `settings.py` |
| `service/` | 2 | 196 | 会话持久化、历史检索、证据缓存 | `store.py` |
| `util/` | 5 | 669 | 文本归一化与清洗、中文分词（bigram）、哈希、token 估算 | `sanitize.py`、`tokenize_zh.py` |
| `prompts/` | 3 | 542 | 提示词与哨兵，**独立可版本化**，中英双语 | `library.py`、`sentinels.py` |
| 顶层 | 4 | 892 | `api.py`（Python API）、`cli.py`（命令行）、`factory.py`（装配） | `factory.py` |

---

## 数据流

### 摄入链路（`stc index`）

```
PDF ──► parser ──► 元数据补全 ──► chunking ──► embedding ──► ┌ 向量索引 ┐
        (pypdf)   (Crossref/      (按章节+      (本地模型)      └ 全文索引 ┘
                   S2/OpenAlex)    句边界)                     (tantivy)
```

- **元数据一等公民**：入库时并发补全三个来源，单源失败自动降级，不阻塞摄入。
- **文件级错误隔离**：单篇论文解析或写入失败**只跳过该篇**，不影响整批
  （`tests/test_ingest_fault_isolation.py` 用 7 条故障注入测试钉住这条）。
- **配置指纹决定索引名**：分块参数、嵌入模型等任一变化 → 指纹变化 → 新索引目录，
  避免"换了模型却读到旧向量"。

### 问答链路（`stc ask`）

```
question ──► retriever ──► screener ──► evidence ──► synthesizer ──► answer + citations
             (四路可插拔)   (两种后端)    (带引用键)    (逐句引用)
```

- **拒答是独立终态**（`REFUSED`），与"模型答了但没引用"（`UNCITED`）分开——
  混为一谈会把"证据不足"和"提示词没被遵守"这两种完全不同的失效糊在一起。
- **引用可回溯率**要求 100%：答案里出现的引用键必须都能解析到真实证据
  （`citation_resolution_rate`）。

### Agentic 链路（`stc ask --mode agentic`）

```
        ┌─────────────────────────────────────────────────┐
        │  agent runtime（自研循环）                        │
        │                                                 │
   ┌────▼─────────────┐  五个工具：                       │
   │ LLM 决策          │   search_literature  （缩小论文范围）│
   │ （选工具 + 参数）  │   gather_evidence    （检索→筛选）  │
   └────┬─────────────┘   answer_question    （合成，不结束）│
        │                 reset_scope        （重置范围）    │
        │                 finish             （结束会话）    │
        │                                                 │
        └──► 四道兜底：超时 / 预算闸门 / 步数上限 / 连续无新证据 ┘
```

`--mode deterministic` 走固定的
`search_literature → gather_evidence → answer_question → finish`，
**字节级可复现**，用作回归测试与消融基线。

---

## 从零开始的阅读与学习路径

> 这份代码库有 12,408 行实现 + 10,662 行测试 + 4,800 行文档。
> 按下面的顺序读，比从 `cli.py` 一头扎进去要省很多时间。

### 路径 A：只想先用起来（约 30 分钟）

1. 本文档下面的[快速开始](#快速开始) —— 装好、建索引、问一个问题
2. [分层架构](#分层架构核心) —— 建立一张地图
3. `stc config show` —— 看清有哪些可调项，比读代码快
4. 遇到"为什么答成这样"，用 `stc ask --json` 看结构化输出，用 `--trace` 看 agent 每一步

**不用读源码也能用。** 想调优再看路径 B。

### 路径 B：想理解架构（约半天）

按**由外向内、由静到动**的顺序读：

| 顺序 | 读什么 | 为什么是这个位置 | 读完能回答 |
| --- | --- | --- | --- |
| 1 | `docs/SPEC.md` §1（质量目标） | 先知道"什么东西算对"，否则后面所有设计都看不出动机 | 引用可回溯率为什么要求 100%？拒答为什么是独立终态？ |
| 2 | `src/scitrace/domain/` | 数据模型是全系统的**词汇表**，不读它后面每个函数签名都要猜 | `Evidence` 与 `Fragment` 差在哪？`Usage.cost_known` 有什么用？ |
| 3 | `src/scitrace/ports/` | 接口定义了"系统能做什么"，实现只是"怎么做" | 换一个 PDF 解析器要动哪些地方？ |
| 4 | `src/scitrace/pipeline/retrieval.py` | 四路检索策略，是理解"可插拔"的最短路径 | `dense_mmr` 与 `hybrid_rrf` 的取舍是什么？ |
| 5 | `src/scitrace/pipeline/screening.py` | 成本大头在这里（逐片段 LLM 打分 + 摘要） | 一次问答的钱花在哪？ |
| 6 | `src/scitrace/pipeline/synthesis.py` | 引用绑定与拒答判定的落点 | 答案里的 `ev-xxxx` 是怎么变成可读引用的？ |
| 7 | `src/scitrace/agent/runtime.py` | Agentic 循环、兜底、历史压缩 | 循环什么时候停？超时了为什么还能拿到答案？ |
| 8 | `src/scitrace/factory.py` | **唯一的装配点**，看完它整个系统的接线就清楚了 | 谁在什么时候决定用哪个实现？ |
| 9 | `src/scitrace/cli.py` / `api.py` | 入口，确认前面的理解 | —— |

### 路径 C：想改代码（几天，配合测试）

1. 先跑 `pytest`，确认 1,150 条测试全绿，**建立"改动可验证"的信心**
2. 读 `tests/test_layering.py` —— 它用 AST 强制分层，**你可能在无意中违反它**
3. 找到你要改的模块对应的测试文件，先读测试再读实现：
   `pipeline/*` ↔ `tests/test_{retrieval,screening,synthesis,ingest,chunking}.py`；
   `agent/*` ↔ `tests/test_agent_runtime.py`（52 条）
4. 改完必须跑 `tools/similarity_audit.py`，**最长公共 token 连续片段必须保持 0**

### 路径 D：想理解"这个项目踩过哪些坑"（最有价值的一条）

`docs/EXPERIMENTS.md`（1,673 行，30 个实验）不是成绩单，更像**一份失败记录**。
建议按下面的分类读，而不是从头顺读：

| 类别 | 实验 | 读它能学到什么 |
| --- | --- | --- |
| **真实缺陷** | 三（真实语料暴露的 2 个崩溃）、六（agentic 首轮的 3 个缺陷）、十三（指标恒为 0）、二十五（禁止词标注一半无效） | 单元测试覆盖不到的失效长什么样 |
| **结论被推翻** | 一（查询集难度翻转结论符号）、十六（测试集调参产生不可复现的改进）、二十一（一个改动修好目标却摧毁别处）、二十四（文献方法移植失败）、二十八（指标与语言耦合） | **测量本身会骗人**，以及怎么发现 |
| **最终成立** | 二十三（去重 −14.4%）、二十九（最终验收） | 什么样的改进才是真的 |

四类测量陷阱已在实验二十八归纳成表：

| 陷阱 | 表现 | 实例 |
| --- | --- | --- |
| 查询集太简单 | 给出**错误符号** | 实验一 |
| 指标结构性恒为 0 | 给出**虚假的好消息** | 实验十三 |
| 在测试集上调参 | 给出**不可复现的改进** | 实验十六 |
| 判据与语言耦合 | 把**语言差异误读成质量差异** | 实验二十八 |

共同的解法只有一句：**动手之前先问"这个数字有没有可能不是它看起来的样子"。**

### 六个"早点知道会省时间"的设计点

1. **`domain/` 零依赖是刻意的**，不是恰好。它让 20 个适配器实现与 5 个管线阶段
   共享同一套词汇而不互相污染。
2. **路径字段不做 NFKC 归一化**。全角字符在文件名里是真实存在的，归一化会让
   "文件找不到"变成一个极难定位的问题。只有文本内容才归一化。
3. **`anyio` task group 里，worker 必须整体包在 try 里**。任何一个未捕获的子任务异常
   会取消所有兄弟任务——一次 36 篇的索引可能因为第 36 篇而全部作废（实验三 3.2）。
4. **`Usage.cost_known` 区分"成本是 0"与"成本未知"**。没有它，一个计价表缺项的模型
   会让成本闸门永远不触发——一个看起来在工作、实际完全失效的闸门。
5. **`REFUSED` 与 `UNCITED` 必须分开**。"证据不足"是正确行为，"答了却没引用"是缺陷，
   合并成一个状态就再也分不清了。
6. **跨配置比较一律看 tokens，不看金额**。金额会被计价表、币种、缓存命中价、
   计价表缺项任意一层弄错——本项目为此先后出过三类错误（实验二十六）。

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

# 4) Agentic 模式（由模型决定检索与取证顺序）
stc ask --mode agentic "两篇论文的方法有何不同？"
stc ask --mode agentic --trace "…"      # 人类可读的逐步追踪
stc ask --json "…"                      # 结构化输出，含 actions 与用量
```

`my_papers/` 需自备 PDF 语料（仓库不含，见 `.gitignore`）。
本项目开发时用的是 36 篇 RAG / Agent / 基础模型方向的论文。

---

## 验证与质量保障

```bash
pytest                                   # 1,150 条测试
pytest --cov=scitrace                    # 覆盖率 92%
python tools/similarity_audit.py \
    --target . --reference ../PaperQA \
    --report docs/AUDIT.md               # 反抄袭审计（5/5 通过）
```

| 手段 | 规模 / 结果 |
| --- | --- |
| 单元 + 集成测试 | **1,150 条通过**，覆盖率 **92%** |
| 架构守卫 | `tests/test_layering.py` 用 AST 强制分层，违反即失败 |
| 行为契约 | SPEC §8 的 C1–C8，逐条对应测试 |
| 真实适配器集成 | 真实 pypdf / tantivy / numpy + 内存替换嵌入与 LLM |
| 故障注入 | `tests/test_ingest_fault_isolation.py`（7 条） |
| 反抄袭审计 | 5 项检查，**最长公共 token 连续片段 = 0** |

**审计工具自身的对抗验证**：从上游植入真实代码后审计必须失败，否则说明它已退化为恒通过。

> 说明：真实 LLM 与本地嵌入模型的端到端运行需要 API key 与模型下载，
> 未在本仓库内执行；端到端链路由上述集成测试覆盖。

---

## 实验体系与如何读它

`benchmarks/` 下有四个脚本：

| 脚本 | 用途 |
| --- | --- |
| `qa_eval.py` | 评测骨架。支持 `--mode`、`--ids`（子集）、`--repeat N`（重复）。**`--repeat` 是必需的**：同题同配置的 LLM 调用次数实测可在 30–87 之间浮动，单次运行只是随机落点 |
| `ablation.py` | 检索策略消融 |
| `screening_ablation.py` | 筛选后端对照 |
| `boundary_corpus.py` | 边界语料（混合格式、超短小节等） |

评测集 `benchmarks/data/rag_qa.jsonl`：**36 题**（28 可答 + 8 库外），
刻意不从任何公开基准复制。关键字段：

- `expected_keywords`：可答题的判据，支持 `|` 分隔的**多语言变体**
  （`"refinement|精炼|refine"`）——因为答案语言会在运行间漂移
- `forbidden_keywords`：库外题的防幻觉判据。**必须是语料中不存在的串**，
  且不得是裸短数字（否则会偶然命中，产生假阳性幻觉）

报告中的 `answer_language_mix` 是必读项：**覆盖率必须与答案语言一起读**，
否则会把语言差异误读成质量差异（实验二十八）。

---

## 实现状态与已知边界

### 已完成

| 增量 | 状态 | 落点 |
| --- | --- | --- |
| ① 自研 Agent 运行时 | 已实现 | `src/scitrace/agent/`（工具协议、循环、状态机、超时兜底、参数容错） |
| ② 可插拔检索内核 + 消融 | 已实现 | `pipeline/retrieval.py` 四路策略；`pipeline/screening.py` 两种后端 |
| ③ 中文文献适配 | 已实现 | `util/tokenize_zh.py`、`pipeline/chunking.py`、中英双语提示词 |
| ④ 成本与延迟治理 | 已实现 | `agent/budget.py`、`Usage`、模型分级路由、`estimate_tokens` |
| ⑤ 自主评测体系 | 已实现 | `benchmarks/qa_eval.py`，30 个实验记录在 `docs/EXPERIMENTS.md` |

### Agentic 模式的实测结果

在 36 题评测集上（9 题调参集 + 27 题留出集）：

| 指标 | 基线 | **当前默认** |
| --- | --- | --- |
| 行为正确率（留出集） | 0.963 | **1.000** |
| 答案覆盖（留出集） | 0.857 | **0.952** |
| 幻觉率 | 0.000 | **0.000** |
| 每题 tokens | 54,418 | **40,161（−26.2%）** |
| 调参集 tokens | 59,167 | **39,568（−33.1%）** |

**质量差异在噪声内**（行为正确率在两批题上方向相反、均为 27 次里 1 次），
成本降 26–33%。降幅来自两件朴素的事：**消除"被筛掉的片段每轮重筛"这一不产生任何
价值的调用**，以及**给筛选摘要加篇幅上限**。

两个负结果已回退到安全默认：`agent.evidence_budget=None`（证据集封顶会让覆盖率
−24pp）、`agent.zero_evidence_fallback=False`（零证据回退省不了钱反而 +11% tokens）。

### 已知局限（诚实声明）

- **语言变体覆盖不完整**：只为已实测出语言敏感的 5 道题手写了中文变体，其余可答题
  可能有同样问题但未测。
- **答案语言会漂移**：英文问题有时被答成中文（留出集 27 题里 10 道）。本项目修的是
  **度量**，未改**行为**——是否强制答案跟随提问语言是产品决定。
- **关键词判据是代理指标**：语义正确但措辞不同的答案会被判未命中，因此
  "答案覆盖"是**下界**而非真值。
- **成本为按自建价目表的估算**，未与账单核对。
- **语料单一**：全部实验基于 36 篇英文论文，换语料后结论是否成立未验证。
- **评测集规模小**：36 题，每层只有 1–3 题，层内结论不能外推。

---

## 开发约定

1. **一次只改一项** —— 否则无法归因。本项目的 k 实验正是靠单变量才定位到取舍曲线。
2. **每项改动都要有反向验证有效的回归测试** —— 临时还原旧行为必须让测试失败；
   否则那条测试是装饰。
3. **报告必须同时给调参集与留出集**，只报一个视为未完成。
4. **调参只在调参集做**，留出集只用于最终确认。
5. **改动实现后必跑反抄袭审计**，最长公共 token 连续片段必须为 0。
6. **文档与代码一起改** —— `docs/SPEC.md` 是冻结规格，改行为要先改规格。

---

## 许可

Apache License 2.0，Copyright 2026 wby。见 `LICENSE` 与 `NOTICE`。

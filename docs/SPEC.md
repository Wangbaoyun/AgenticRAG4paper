# scitrace 行为规格（SPEC）

| 项 | 值 |
| --- | --- |
| 版本 | `spec-v1`（冻结） |
| 冻结日期 | 2026-02（见 git 历史中本文件的首次提交） |
| 状态 | **冻结基线**。实现阶段只以本文件为依据；变更须走 §11 的变更流程 |
| 适用对象 | `src/scitrace/**` 的全部实现与 `tests/**` |

---

## 0. 本文件的地位与洁净室纪律

本文件是 `scitrace` 的**唯一实现依据**。其内容来源于：

1. PaperQA2 的**公开论文**（arXiv:2409.13740）与公开文档所述的方法；
2. 对本系统**自身**的设计决策（下述各节中的具体参数、命名、算法均为本项目自定）；
3. **黑盒行为观察**——运行参考实现、观察其输入输出行为（不含阅读其源码实现）。

**纪律（违反即审计失败）**：

- ❌ 不得将上游的源代码、Prompt 字符串、注释、docstring、文档文本复制或"改写式搬运"进本项目；
- ❌ 不得引入 `paperqa` / `paper-qa*` / `fhaviary` / `aviary` / `fhlmi` / `ldp` 作为依赖；
- ❌ 不得使用 `PaperQA` / `pqa` / `pqac` 等上游标识符作为本项目命名；
- ❌ 不得搬运上游的测试数据资产（`tests/stub_data/*`、`tests/cassettes/*.yaml`、
  `docs/2024-10-16_litqa2-splits.json5`）；
- ✅ 允许使用论文与文档所述的**方法、流程、工程思想**（思想/表达二分法）。

污染事件与流程记录见 `docs/PROVENANCE.md`；机械化验证见 `tools/similarity_audit.py` 与 `docs/AUDIT.md`。

---

## 1. 系统目标与非目标

### 1.1 目标

给定一批本地 PDF 文献与一个自然语言问题，产出：

1. 一个**基于文献内容**的答案；
2. 答案中每一处论断带有**可回溯到原文片段**的引用键；
3. 一份**参考文献列表**（BibTeX 可导出）；
4. 当文献不足以回答时，**明确拒答**而非编造。

### 1.2 非目标（v1 明确不做）

- 多模态富化（论文插图/表格的 VLM 描述）——**首版不做**，但在 `ParsedFragment` 中预留 `media` 字段；
- ClinicalTrials.gov 等特定领域数据源工具；
- Zotero / OpenReview 等第三方文献管理器集成；
- Web UI（仅 CLI）；
- 分布式索引（单机内存 + 本地磁盘）。

### 1.3 质量目标（可验收）

| 指标 | 目标 |
| --- | --- |
| 引用可回溯率 | 答案中出现的引用键 100% 能解析到实际片段 |
| 拒答正确性 | 对语料外问题，输出 `REFUSED` 状态且不产生引用 |
| 可复现性 | 确定性模式下，同输入同配置 → 字节级相同输出 |
| 索引复用 | 未变更语料二次 `index` 不触发重新解析/嵌入 |
| 成本可观测 | 每次会话输出 token 用量与估算成本 |

---

## 2. 领域术语与数据流

### 2.1 术语表（本项目自定命名）

| 术语 | 含义 | 对应概念（仅供对照，非命名依据） |
| --- | --- | --- |
| `Source` | 一篇文献：稳定标识 + 书目元数据 | 文献实体 |
| `SourceKey` | 文献的稳定键，`sha256(doi or content_hash)[:16]` | 文档键 |
| `Fragment` | 文献切分出的一个文本块 | 文本块 |
| `Evidence` | 经过相关性筛选的片段 + 摘要 + 0–10 分 | 证据上下文 |
| `EvidenceKey` | 证据的引用键，形如 `ev-<8 hex>` | 引用键 |
| `Session` | 一次问答的完整状态：问题、答案、证据、用量 | 问答会话 |
| `IndexFingerprint` | 由配置派生的索引标识，12 位 hex | 索引名哈希 |

### 2.2 端到端数据流

```
PDF/文本
  │ ① parse
  ▼
ParsedDocument{pages, full_text, hints}
  │ ② chunk（结构感知）
  ▼
Fragment[]  ──③ embed──►  向量
  │                        │
  │ ④ index                │
  ▼                        ▼
FullTextIndex          VectorIndex          （共享 IndexFingerprint 目录）
  │                        │
  └──────⑤ retrieve────────┘
              │
              ▼
        Fragment[] (候选)
              │ ⑥ screen（可插拔：LLM-RCS / Cross-Encoder）
              ▼
        Evidence[]  ──► ⑦ synthesize ──► Answer{text, citations, references}
                              │
                              ▼
                        Session（持久化）
```

---

## 3. 阶段规格

### 3.1 摄入（Ingest）

**输入**：文件路径或目录路径列表。
**输出**：`IngestReport{added, skipped, failed, removed, sources, fragments}`。

规则：

- **支持格式**：`.pdf`（v1 必须）、`.txt`、`.md`。遇到不支持格式记录为 `skipped`，不中断。
- **递归**：目录递归扫描；跳过隐藏文件与 `.git/`。
- **文件哈希**：对整个文件内容取 `sha256`，16 位 hex 作为 `content_hash`。
- **唯一性**：`SourceKey = sha256(doi)[:16] if doi else sha256(content_hash)[:16]`。
  同一 `SourceKey` 再次摄入视为**更新**（覆盖元数据、重建片段）。
- **清单（manifest）**：`<index_dir>/manifest.json`，结构
  `{ "schema": 1, "files": { "<rel_path>": {"hash","source_key","status","indexed_at","error"} } }`。
  `status ∈ {ok, failed, skipped}`。
- **增量语义**：`index` 时
  - `hash` 未变且 `status == ok` → `skipped`，不重新解析/嵌入；
  - `hash` 变化 → 删除旧片段后重新摄入；
  - 清单中有、磁盘上不存在 → `removed`，同步从双索引删除；
  - `status == failed` → 本次重试（即使 hash 未变）。
- **原子性**：单文件失败不得影响其他文件；失败写入 `status=failed` + `error` 后继续。
  全部失败时命令退出码为 1。
- **旧片段清理**：`hash` 变化时**必须**先按 `source_key` 移除该文件的旧片段再写入新片段
  （因 `fragment_id` 含内容哈希，旧 id 不会被新片段覆盖）。
  **例外**：若另一个清单项共享同一个 `source_key`（同一 DOI 的预印本与正式版），
  则跳过清理——`remove(source_key)` 会连对方的片段一起删除。
  代价是该文件自己的旧片段可能残留。这是**已知且刻意接受的取舍**：
  索引的删除单位是 `source_key`，要做到文件级精确删除需端口支持按 `fragment_id` 删除。
  共用 `source_key` 的情形会计入 `report.duplicate_sources` 并在 CLI 中提示用户。

### 3.2 分块（Chunking）

采用**三级结构感知分块**（本项目自定，非上游策略）：

1. **结构切分**：按文档结构（PDF 的字体大小/行距启发式，或 Markdown 标题）切出 section，
   记录 `section_path`（如 `["2 Method", "2.1 Retrieval"]`）。
2. **段落聚合**：在 section 内按空行/缩进切段落；相邻段落合并直到达到 `target_chars`。
3. **超长切分**：单段超过 `max_chars` 时，在**句子边界**切分，保留 `overlap_chars` 重叠。
   - 中文句末标点：`。！？；…`
   - 英文句末标点：`. ! ? ;`（需排除 `et al.`、`Fig.`、`e.g.` 等常见缩写，内置缩写表）

参数（`ingest.chunking`）：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `target_chars` | 1500 | 段落聚合目标长度 |
| `max_chars` | 3000 | 单块硬上限，超出按句切分 |
| `min_chars` | 300 | 低于此值的块与相邻同 section 块合并 |
| `overlap_chars` | 200 | 超长切分时的句级重叠 |
| `drop_references` | `true` | 丢弃文末参考文献列表区块（避免引用噪声） |

每个 `Fragment` 记录：`fragment_id`（`sha256(source_key + document_hash + section_path + chunk_index)[:16]`）、
`source_key`、`text`、`chunk_index`、`page_range`、`section_path`、`char_count`、`media=[]`。

> **v1.1 变更**：`fragment_id` 原先不含 `document_hash`，理由是"同一位置重新切分会得到不同
> `chunk_index`，同一篇论文重新解析会得到相同 id"。该推理漏掉两种真实情形：
> ① 同一 DOI 的两个文件（预印本 / 正式版）共享 `source_key`，其 `(section_path, chunk_index)`
> 序列高度相似，会产生**相同 id** 而互相覆盖，索引里留下来源混杂的文档；
> ② 同一 DOI 而内容被修订（用户替换勘误版 PDF）时 `source_key` 不变，旧片段不会被识别为陈旧。
> 加入内容哈希后，内容相同仍得到相同 id（幂等去重能力保留），内容不同则必然不同。
> 代价是"内容变化后须显式清理旧片段"，由 §3.1 的摄入流程按 `source_key` 移除来完成。

### 3.3 向量化（Embedding）

- 端口：`EmbeddingClient.embed(texts: Sequence[str], *, kind: Literal["query","document"]) -> Sequence[Vector]`。
- `kind` 用于支持"查询/文档"非对称编码模型。
- **归一化**：返回向量必须 L2 归一化；相似度即点积（余弦）。
- 默认后端：本地 `sentence-transformers`（配置 `embedding.model`，默认中文友好模型）。
- 备选后端：OpenAI 兼容 `/v1/embeddings` 端点。
- **批处理**：按 `embedding.batch_size`（默认 32）分批；失败重试 3 次（指数退避）。
- **缓存**：`sha256(model_name + text)[:32] -> vector`，落盘 `<index_dir>/embeddings.npz`，
  命中的文本不重复调用模型。

### 3.4 双索引（Dual Index）

两个索引共存于 `<index_dir>`（由 §5.3 的指纹决定），互补使用：

**A. 向量索引 `VectorIndex`**

- 端口方法：`add(fragments)`、`search(query_vec, k, allowed_keys=None)`、
  `mmr_search(query_vec, k, fetch_k, lambda_)`、`remove(source_key)`、`clear()`、`__len__`。
- 实现 `NumpyVectorIndex`：内存 numpy 矩阵 + id 列表；持久化 `vectors.npy` + `fragments.jsonl`。
- `allowed_keys` 用于"在已选定的论文范围内检索"（Agent 的 `paper_search` 之后）。

**B. 全文索引 `FullTextIndex`**

- 端口方法：`add(fragments)`、`search(query, k, allowed_keys=None)`、
  `remove(source_key)`、`commit()`、`__len__`。
- 实现基于 `tantivy`，schema：
  `fragment_id`(STRING|STORED)、`source_key`(STRING|STORED)、`text_zh`(TEXT)、
  `text_en`(TEXT)、`title`(TEXT|STORED)。
- **中文处理（原创设计）**：不引入分词依赖。入库前把中文文本转成**字符二元组
  （bigram）序列**，以空格连接写入 `text_zh`，字段 tokenizer 用 `whitespace`；
  查询侧同法转换。英文原样写入 `text_en`，tokenizer 用 `en_stem`。
  两字段查询结果按 `max(score_zh, score_en)` 合并。
- **提交语义**：`add` 后必须 `commit()` 才对 `search` 可见。

### 3.5 元数据补全（Metadata）

**端口**：`MetadataProvider.lookup(patch: SourcePatch) -> SourcePatch | None`，
`SourcePatch` 为可选字段的稀疏结构（title / authors / year / doi / venue / citation_count /
is_oa / oa_url / quality_tier / retracted）。

**Provider 列表（v1）**：

| Provider | 端点 | 主要贡献字段 | 触发条件 |
| --- | --- | --- | --- |
| Crossref | `api.crossref.org/works/{doi}`、`/works?query.bibliographic=` | title, authors, year, venue, doi | 有 DOI 或标题 |
| Semantic Scholar | `api.semanticscholar.org/graph/v1/paper/search` | citation_count, venue, externalIds | 有标题 |
| OpenAlex | `api.openalex.org/works` | is_oa, oa_url, citation_count, quality_tier | 有 DOI 或标题 |
| RetractionWatch 快照 | 本地 CSV（可选配置） | retracted | 有 DOI |

**合并策略（本项目自定）**：

- **字段级优先级**：`doi` 校验通过的结果 > 标题模糊匹配（`rapidfuzz`-free 自实现
  token Jaccard ≥ 0.8）的结果；
- 书目字段（title/authors/year/venue）优先 Crossref；
  `citation_count` 优先 Semantic Scholar；
  `is_oa`/`oa_url` 优先 Unpaywall/OpenAlex；
  `quality_tier` 由 OpenAlex 的 `host_venue` 分位映射为 0–3；
- **不允许覆盖**已有非空字段，除非新值来自更高优先级来源。

**降级与容错**：

- 每个 provider 独立超时（默认 15s）、重试 3 次（指数退避 + 抖动）；
- 任一 provider 失败 → 记录 warning，返回 `None`，**不影响**主流程；
- **全部** provider 失败 → 仍完成摄入，`Source` 只有本地可推导字段（title 取 PDF 内嵌元数据
  或文件名）；
- 元数据补全过程与解析/嵌入**并发**执行，不串行阻塞。

### 3.6 检索（Retrieval）—— 可插拔内核【原创增量 ②】

`retrieval.strategy` 四选一，统一返回 `Fragment[]`（长度 ≤ `k`）：

| 策略 | 行为 |
| --- | --- |
| `dense` | 向量相似度 top-k |
| `dense_mmr` | 先取 `fetch_k`（默认 `4*k`），再做 MMR 去冗余到 k，`lambda_` 默认 0.5 |
| `hybrid_rrf` | 向量 top-`k` 与 BM25 top-`k` 两路结果做 **RRF 融合**（`rrf_k=60`），取前 k |
| `hybrid_rrf_rerank` | 在 `hybrid_rrf` 的候选（`rerank_pool`，默认 `3*k`）上做 Cross-Encoder 重排，取前 k |

- **MMR 定义**（本项目自实现）：`score = λ·sim(q,d) − (1−λ)·max_{s∈S} sim(d,s)`，贪心选点；
  `sim` 为归一化向量点积。
- **RRF 定义**：`score(d) = Σ_r 1/(rrf_k + rank_r(d))`，rank 从 1 开始；缺失路不计分。
- `rerank` 后端缺失（未安装可选依赖）时，配置为 `hybrid_rrf_rerank` 必须**显式报错**，
  不得静默降级（避免实验数据不可信）。

### 3.7 证据筛选（Screening）

**端口**：`EvidenceScreener.screen(question: str, fragments: Sequence[Fragment]) -> Sequence[Evidence]`。

两种实现，构成**消融轴**：

**A. `LLMScreener`（默认，RCS 等价物）**

- 对每个片段调用 LLM，要求返回严格 JSON：`{"summary": str, "relevance_score": int 1..10}`；
- 若片段与问题无关，要求 `summary` 返回固定的"不适用"标记（本项目自定英文哨兵
  `NOT_APPLICABLE`，与任何上游表述无关）；
- **容错解析**：先直接 `json.loads`；失败则依次尝试 ① 剥离 markdown 代码围栏
  ② 剥离思维链标签 ③ 用括号配平截取最外层对象 ④ 从文本中正则提取
  `relevance_score` 与其余内容作为 summary。仍失败 → 该片段记为 `relevance=0` 并计一次
  `parse_failure` 指标（**不得**让单块解析失败中断整批）；
- `relevance_score` 若为 `"8/10"` 形式或浮点，归一化为 `1..10` 整数，越界钳制；
- **并发**：`screening.concurrency`（默认 8）；
- **过滤**：`relevance >= screening.min_relevance`（默认 5）进入证据集；
- **排序**：按 `relevance` 降序，取前 `answer_max_evidence`（默认 10）。

**B. `CrossEncoderScreener`（对照实现）**

- 用 Cross-Encoder 对 `(question, fragment)` 打分，取前 `presummary_n`（默认 15）；
- 仅对这 15 个片段调用 LLM 生成**摘要**（不要求打分），成本显著下降；
- 用于回答"LLM 重排相对 Cross-Encoder 的边际收益是多少"。

**EvidenceKey 生成**：`"ev-" + sha256(f"{source_key}:{fragment_id}")[:8]`，**确定性**，
同一证据在任何运行中键相同。

### 3.8 答案合成（Synthesis）

**输入**：`question`、`Evidence[]`、可选 `prior_answer`（迭代场景）。
**输出**：`Answer{text, evidence_keys_used, references, refused: bool}`。

规则：

1. **上下文组装**：每条证据渲染为
   `[{evidence_key}] {citation_text}\n{summary}`，用空行分隔，整体置于提示词中。
   `citation_text` 形如 `(author2024title pages 3-4)`，由 `Source` 元数据生成；
   元数据缺失时退化为 `(unknown pages 3-4)` 并计入 `metadata_incomplete` 指标。
2. **引用约束**：提示词中明确列出**合法引用写法**与**非法写法示例**（示例用本项目自己的
   虚构键，不得与上游示例雷同）：合法为 `(ev-1a2b3c4d)` 或
   `(ev-1a2b3c4d, ev-5e6f7a8b)`；非法包括 `and`/`;` 连接、连字符范围、以作者名开头等。
3. **拒答**：提示词要求证据不足时回复固定哨兵短语（本项目自定：
   `INSUFFICIENT_EVIDENCE`，中文场景为「证据不足，无法回答」）。命中哨兵 → `refused=True`。
4. **引用后处理**：正则提取答案中的 `ev-<8hex>`；**只保留**能解析到本次证据集的键，
   未知键剥离并计入 `dangling_citation` 指标（该指标 > 0 时应告警）。
5. **引用渲染**：正文中的 `ev-<8hex>` 替换为最终形如 `(author2024title pages 3-4)` 的
   文内引用；参考文献列表按首次出现顺序去重生成。
6. **迭代**：提供 `prior_answer` 时，新答案**只能**使用本次上下文中出现的证据键。

### 3.9 引用与参考文献

- **BibTeX**：由 `Source` 元数据生成，key = `firstauthorlastname + year + firstsignificanttitleword`
  （小写、去非字母数字）。字段缺失时省略该字段，不产生空字段。
- 生成后必须能被 `pybtex` 解析通过；解析失败 → 该条目标记为 `bibtex_invalid` 并回退为
  仅含 title/year 的最小条目。
- 会话结束输出 `references: dict[evidence_key_prefix -> bibtex]`（按首次引用顺序）。

### 3.10 错误与边界行为

| 场景 | 规定行为 |
| --- | --- |
| 空语料 / 索引为空 | `ask` 立即返回 `REFUSED`，提示"索引为空"，不调用 LLM |
| 索引不存在 | 报错并提示先运行 `stc index`，退出码 1 |
| PDF 解析失败 | 该文件 `status=failed`，其余继续；`index` 结束打印失败清单 |
| LLM 调用失败 | 重试 3 次后：筛选阶段跳过该片段；合成阶段 → 状态 `FAIL`，退出码 1 |
| 响应非 JSON | 按 §3.7 容错链降级；全部失败不影响整批 |
| 超长文档（>2000 页） | 分页流式解析，峰值内存不得随页数线性增长 |
| 编码异常字符 | 替换为 U+FFFD 后继续，不抛异常 |

---

## 4. Agent 运行时【原创增量 ①】

**完全自研**：不依赖任何第三方 Agent/环境框架。核心为约 600 行的 `agent/` 包。

### 4.1 工具集（本项目自定命名与语义）

| 工具 | 签名 | 语义 |
| --- | --- | --- |
| `search_literature` | `(query: str, year_from?: int, year_to?: int) -> str` | 在全文/向量索引中检索**论文**，更新"候选论文集"，返回候选清单与计数 |
| `gather_evidence` | `(question: str) -> str` | 在候选论文范围内检索片段 → 筛选 → 追加到证据集，返回新增证据数与状态摘要 |
| `answer_question` | `() -> str` | 基于当前证据集合成答案，返回答案文本（不结束会话，允许追问式迭代） |
| `reset_scope` | `() -> str` | 清空候选论文集（保留证据集） |
| `finish` | `(has_answer: bool, reason: str) -> None` | 结束会话。`has_answer=false` 时状态为 `UNSURE` |

- 工具参数使用 JSON Schema 描述（自研的 `Tool` 协议 + pydantic 模型导出 schema）。
- **参数解析容错**：模型输出的工具参数若不是合法 JSON，先尝试修复（补齐引号/去除尾逗号/
  提取首个对象），仍失败则**回灌错误信息给模型重试**，最多 5 次；超过则终止并置 `FAIL`。

### 4.2 循环与状态机

```
IDLE ──reset──► RUNNING ──finish(has_answer=true)──► SUCCESS
                   │  └──finish(has_answer=false)──► UNSURE
                   ├──超时 / 超出步数 / 超出预算────► TRUNCATED（强制合成答案）
                   └──不可恢复错误────────────────► FAIL
```

- `max_steps` 默认 12（一次 LLM 决策 + 工具执行 = 1 步）；
- 每步把工具观测结果追加进消息历史；历史超过 `context_token_limit`（默认 32k）时，
  对**旧观测**做有损压缩（只保留工具名 + 摘要行），保留首尾；
- **状态摘要**：每次工具返回都附带一行状态串
  `papers={total}/{relevant} evidence={count} cost=${x}`，注入下一轮提示。

### 4.3 兜底策略

- **超时兜底**：`agent.timeout_seconds`（默认 500）到期 → 立即用**已有证据**强制合成答案，
  状态 `TRUNCATED`；
- **预算闸门**【原创增量 ④】：`agent.max_tokens` / `agent.max_cost` 触顶 → 同上，
  并在返回值中标注 `budget_exceeded`；
- **步数上限**：达到 `max_steps` → 强制合成，状态 `TRUNCATED`；
- **证据为空即结束**：若连续 2 次 `gather_evidence` 未新增证据，自动执行 `answer_question`
  后 `finish`。

### 4.4 确定性模式

`--mode deterministic`：不调用 LLM 做工具选择，固定执行
`search_literature(question) → gather_evidence(question) → answer_question() → finish(true)`。
要求：同输入同配置 → 输出字节级可复现。用于回归测试与消融基线。

---

## 5. 配置系统

### 5.1 分层与优先级

`显式参数(CLI) > 环境变量 > 命名配置文件 > 内置默认`

- 环境变量前缀 `SCITRACE_`，嵌套用 `__`（如 `SCITRACE_LLM__MODEL`）；
- 命名配置文件：`~/.scitrace/settings/<name>.json`，`--settings <name>` 选择，默认 `default`；
- 未知字段**一律报错**（`extra="forbid"`），避免拼写错误静默生效。

### 5.2 配置模型分组

`llm` / `pricing` / `embedding` / `ingest` / `retrieval` / `screening` / `answer` / `agent` / `metadata` / `index`。

**`pricing`（v1.3 新增）**：自备计价表（币种 + 输入/缓存命中/输出每百万 token 单价）。

存在的理由很具体：litellm 的价格表**不收录**自建或代理的模型名（实测 `deepseek-v4-flash`
不在其中）。缺失的后果不是"成本显示不出来"，而是 `estimated_cost` 恒为 0 →
**成本闸门永远不触发**——一个看起来在工作、实际完全失效的闸门。

币种是显式字段而非隐含约定：本项目面向中文语料，供应商多以人民币计价，
把人民币数字写进名叫 `_usd` 的字段是错的。**`Usage.estimated_cost_usd` 因此更名为
`estimated_cost`，并新增 `cost_currency`**（v1.3 破坏性变更，`--json` 输出同步调整）。

模型工厂：`get_llm(role: Literal["main","summary","agent"])` 允许三个角色使用**不同模型**，
支持"用便宜模型做筛选、用强模型做合成"的分级路由【原创增量 ④】。

### 5.3 索引指纹

```
IndexFingerprint = sha256(canonical_json({
    "schema": SCHEMA_VERSION,
    "parser": ingest.parser,
    "chunking": ingest.chunking,
    "embedding_backend": embedding.backend,
    "embedding_model": embedding.model,
}))[:12]
```

- 索引目录：`~/.scitrace/index/<fingerprint>/`；
- **指纹只由"表示配置"决定**（解析器、分块、嵌入模型），**不含 `source_dirs`**。

  > 这是对 v1 初稿的一处修正。初稿把语料路径也算进指纹，会导致一个必然踩到的坑：
  > `stc index ./papers` 时 `source_dirs=./papers` 参与指纹，而 `stc ask`
  > 不带路径时 `source_dirs` 为空 → 指纹不同 → 报"索引不存在"。
  > 更根本的问题是概念错位：**索引的同一性应当由"文本如何被表示"决定，
  > 而不是由"这次索引了哪些目录"决定**。
- **语料归属**由 `<index_dir>/manifest.json` 的 `rel_path` 记录，`stc status` 报告
  各语料根及其文献数。需要"只在某个语料内提问"时，走查询侧的语料过滤
  （`allowed_keys`），而不是划分索引目录——后者会让同一篇论文在多语料下被重复解析与嵌入。
- 指纹变化即视为新索引，不自动迁移旧索引（避免"配置变了但索引是旧的"这类静默错误）；
- `stc status` 必须打印当前指纹、索引中的文献数与片段数。

---

## 6. 持久化与会话

- **索引**：`~/.scitrace/index/<fingerprint>/{manifest.json, vectors.npy, fragments.jsonl, fts/, embeddings.npz}`。
- **会话历史**：`~/.scitrace/sessions/<fingerprint>/<session_id>.json`，含问题、答案、
  证据键、引用、token 用量、耗时、状态。
- **历史检索**：`stc sessions search "<关键词>"` 在历史答案上做全文检索。

---

## 7. CLI 契约

| 命令 | 参数 | 退出码 |
| --- | --- | --- |
| `stc index <path>...` | `--settings`、`--rebuild` | 0 成功 / 1 全部失败 |
| `stc ask "<q>"` | `--settings`、`--mode {agentic,deterministic}`、`--json` | 0 成功（含拒答）/ 1 错误 / 2 用法错误 |
| `stc search "<q>"` | `--settings`、`--k`、`--json` | 0 / 1 / 2 |
| `stc status` | `--settings` | 0 / 1 |
| `stc sessions search "<kw>"` | `--json` | 0 / 1 |
| `stc config {show,save,path,init}` | `--settings` | 0 / 1 / 2 |

`--json` 输出 schema（`ask`）：

```json
{
  "session_id": "…", "question": "…", "answer": "…",
  "status": "SUCCESS|UNSURE|TRUNCATED|FAIL|REFUSED",
  "citations": [{"evidence_key": "ev-1a2b3c4d", "citation": "(author2024title pages 3-4)",
                 "source_key": "…", "page_range": [3, 4], "relevance": 8}],
  "references": {"author2024title": "@article{…}"},
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
            "estimated_cost": 0.0, "cost_currency": "CNY", "cost_known": true},
  "timing": {"retrieve_s": 0.0, "screen_s": 0.0, "synthesize_s": 0.0, "total_s": 0.0}
}
```

---

## 8. 可观测行为契约（对拍与回归的依据）

以下 8 条是与参考实现"逻辑一致"的判定维度，必须由 `tests/conformance/` 覆盖：

| # | 契约 | 断言 |
| --- | --- | --- |
| C1 | 三段式阶段顺序 | 一次成功问答必然经过 检索 → 取证 → 合成 三阶段（可由事件日志验证） |
| C2 | 引用产出 | 成功答案必含 ≥1 个可回溯引用键，且 `references` 非空 |
| C3 | 拒答行为 | 语料外问题 → `REFUSED`，无引用，不产生幻觉性文献 |
| C4 | 评分过滤语义 | `relevance < min_relevance` 的证据**绝不**出现在上下文中 |
| C5 | 索引重建触发 | 仅当指纹变化或文件 hash 变化时重建 |
| C6 | 工具集语义 | 工具名、参数含义、`finish` 的 `has_answer` 语义稳定 |
| C7 | 配置优先级 | 显式参数 > 环境变量 > 配置文件 > 默认，逐层可验证 |
| C8 | 降级路径 | 元数据源失败、单块 LLM 解析失败、单文件解析失败均不中断主流程 |

---

## 9. 原创增量落点（本项目的自有贡献）

| # | 增量 | 落点 | 验收 |
| --- | --- | --- | --- |
| ① | 自研 Agent 运行时 | `agent/`（loop/tools/state/budget） | 无第三方 Agent 框架依赖；工具协议与状态机为自定 |
| ② | 可插拔检索内核 + 消融 | `pipeline/retrieval.py`、`pipeline/screening.py` | 4 种策略可切换；产出对照实验结果表 |
| ③ | 中文科研文献适配 | `util/tokenize_zh.py`（bigram）、`pipeline/chunking.py`（中文句边界）、中文引用格式 | 中文语料检索召回可用；中英混合语料不互相干扰 |
| ④ | 成本与延迟治理 | `agent/budget.py`、`service/cache.py`、模型分级路由 | 可输出 token/成本报告；缓存命中可观测 |
| ⑤ | 自主评测体系 | `benchmarks/`（M8） | 可复现的评测脚本 + 中文自建集 |

> ①②③④ 在 M1–M6 落地架构与骨架；②的消融实验与 ⑤ 的评测执行属 M8，超出首版范围。

---

## 10. 明确禁止清单（审计脚本会检查）

1. 依赖：`paperqa`、`paper-qa*`、`fhaviary`、`aviary`、`fhlmi`、`ldp`；
2. 标识符：`PaperQA`、`pqa`、`pqac`；
3. 文本：任何长度 ≥ 20 字符的字符串字面量与上游相似度 ≥ 0.6；
4. 文件：与上游 git 跟踪文件 SHA-256 相同的资产；
5. 代码：与上游 8-gram 重叠率 > 5%（单文件）。

---

## 11. 变更流程

1. 修改本文件 → 2. 更新 § 版本号 → 3. `docs/PROVENANCE.md` 追加变更理由 →
4. 同步更新对应 `tests/conformance/` 契约 → 5. 重跑审计 → 6. 提交。

冻结的意义在于：实现阶段的偏差由**测试**暴露，而不是由"边写边改规格"掩盖。

### 修订记录

| 版本 | 变更 | 理由 |
| --- | --- | --- |
| v1 | 初稿 | — |
| v1.1 | §5.3 索引指纹**移除** `source_dirs` | 原设计会导致 `stc index ./papers` 后 `stc ask`（不带路径）因指纹不同而报"索引不存在"。更根本的是概念错位：索引同一性应由"文本如何被表示"决定，而非"这次索引了哪些目录" |
| v1.2 | §3.2 `fragment_id` **加入** `document_hash`；§3.1 摄入流程相应改为按 `source_key` 显式清理旧片段 | 见 §3.2 的变更说明 |
| v1.3 | §5.2 新增 `pricing` 配置组；§7 用量字段由 `estimated_cost_usd` 改为 `estimated_cost` + `cost_currency` + `cached_tokens` + `cost_known` | 供应商以人民币计价，把人民币写进 `_usd` 字段是错的；缓存命中价便宜两个数量级，不区分会高估数倍；`cost_known` 区分"成本为 0"与"成本未知" |

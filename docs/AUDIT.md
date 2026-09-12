# 洁净室相似度审计报告（Clean-room Similarity Audit）

> 本报告由 `tools/similarity_audit.py` 自动生成，仅依据文本与结构证据判定，不构成法律意见。

## 元信息

| 项目 | 值 |
| --- | --- |
| 生成时间 | 2026-09-12 15:08:56 +0800 |
| 目标项目（待审计） | `/home/wby/MyPaperQA` |
| 参考实现（比对基准） | `/home/wby/PaperQA` |
| 上游 HEAD commit | `0a03fb09619c566470d07ab443ca8a7376b11d00` |
| 上游 HEAD 提交信息 | chore: 移除不再使用的上游教程与 CI 配置 |
| 参考侧文件集来源 | git ls-files（索引中的跟踪文件） |
| 上游 HEAD 跟踪文件数 | 172（另有 0 个路径在 HEAD 中不存在，已跳过） |
| 目标侧纳入审计的文件数 | 162 |
| n-gram 长度（源码 / 文档） | 8 / 12 |
| 整体 containment 阈值 | 2.00% |
| 单文件 containment 阈值 | 5.00% |
| 字符串相似度阈值 | 0.60 |
| 字符串违规所需匹配字符数 | 32 |
| 逐字复制：实质性行上限 / 对齐片段下限 | 0 / 5 |
| 审计结论 | ✅ 全部通过 |

## 结论摘要

| 检查项 | 状态 | 关键指标 | 阈值 |
| --- | --- | --- | --- |
| 1. Token n-gram 重叠（n=8） | ✅ 通过 | 整体 containment 0.4572%（Jaccard 0.2474%）；单文件最高 13.8298%；最长公共 token 连续片段 0 | 整体 containment ≤ 2.00%；且**任一文件**不得存在 ≥ 25 token 的连续公共片段（该判据不受文件大小豁免）；单文件 containment > 5.00% 且无长连续片段者列为待复核 |
| 2. 逐行精确复制 | ✅ 通过 | 逐字相同行 385（实质性 0 / 通用惯用式 385）；最长对齐片段 2 行 | 实质性 ≤ 0 行 且 对齐片段 < 5 行 |
| 3. 字符串字面量相似度（阈值 0.60） | ✅ 通过 | 违规相似串 0 条（≥0.60 相似串 41 条）；专有标识符违规 0 处（说明性提及 10 处） | 相似度 ≥ 0.60 且匹配字符 ≥ 32 的相似串 = 0 且 代码中的标识符违规 = 0 |
| 4. 依赖审计 | ✅ 通过 | 禁用依赖 0 处；警告 0 处 | 禁用依赖 = 0 |
| 5. 资产 / 文档审计（文档 n=12） | ✅ 通过 | 哈希相同资产 0 个；文档整体 containment 0.0400%（最高 0.6594%） | 哈希相同 = 0 且 文档整体 ≤ 2.00% 且 单文档 ≤ 5.00% |

## 1. Token n-gram 重叠（n=8）

**状态：** ✅ 通过　　**关键指标：** 整体 containment 0.4572%（Jaccard 0.2474%）；单文件最高 13.8298%；最长公共 token 连续片段 0　　**阈值：** 整体 containment ≤ 2.00%；且**任一文件**不得存在 ≥ 25 token 的连续公共片段（该判据不受文件大小豁免）；单文件 containment > 5.00% 且无长连续片段者列为待复核

- 上游参考 n-gram 池：**82,410** 个（来自 58 个 `.py` 文件）
- 目标 n-gram 池：**96,685** 个（扫描 96 个 `.py` 文件）
- 交集：**442** 个

#### 重叠最高的前 20 个目标文件

| # | 目标文件 | containment | 共有 n-gram | 文件 n-gram 总数 | 最长公共连续片段 |
| --- | --- | --- | --- | --- | --- |
| 1 | `src/scitrace/ports/common.py` | 13.8298% | 13 | 94 | 0 |
| 2 | `src/scitrace/ports/fulltext_index.py` | 9.0909% | 14 | 154 | 0 |
| 3 | `src/scitrace/domain/evidence.py` | 8.1218% | 16 | 197 | 0 |
| 4 | `src/scitrace/domain/fragment.py` | 5.7743% | 22 | 381 | 0 |
| 5 | `src/scitrace/ports/llm.py` | 5.5138% | 22 | 399 | 0 |
| 6 | `src/scitrace/domain/source.py` | 4.8876% | 50 | 1,023 | 0 |
| 7 | `src/scitrace/domain/session.py` | 4.7047% | 47 | 999 | 0 |
| 8 | `src/scitrace/config/settings.py` | 4.0894% | 97 | 2,372 | 0 |
| 9 | `src/scitrace/ports/vector_index.py` | 3.4146% | 7 | 205 | 0 |
| 10 | `src/scitrace/adapters/llm/embedding.py` | 3.2787% | 28 | 854 | 0 |
| 11 | `src/scitrace/util/hashing.py` | 2.7778% | 10 | 360 | 0 |
| 12 | `src/scitrace/ports/metadata.py` | 2.7322% | 5 | 183 | 0 |
| 13 | `tests/fakes.py` | 2.4506% | 36 | 1,469 | 0 |
| 14 | `src/scitrace/agent/state.py` | 2.0367% | 10 | 491 | 0 |
| 15 | `src/scitrace/pipeline/synthesis.py` | 1.9069% | 25 | 1,311 | 0 |
| 16 | `src/scitrace/service/store.py` | 1.8182% | 12 | 660 | 0 |
| 17 | `src/scitrace/adapters/metadata/http_base.py` | 1.7801% | 34 | 1,910 | 0 |
| 18 | `src/scitrace/pipeline/retrieval.py` | 1.6310% | 16 | 981 | 0 |
| 19 | `benchmarks/screening_ablation.py` | 1.5598% | 9 | 577 | 0 |
| 20 | `tests/test_util.py` | 1.5544% | 12 | 772 | 0 |

**✅ 统计超标但判定通过的文件（containment > 5.00%，但最长公共片段 < 25 token）：**
- `src/scitrace/domain/fragment.py` → containment 5.7743%，最长公共片段 0 token
- `src/scitrace/ports/llm.py` → containment 5.5138%，最长公共片段 0 token

> 这类文件的重叠来自**框架与领域的趋同**（同用 pydantic 表达同一领域概念时，字段声明、装饰器与类型标注等脚手架序列必然重合），属于著作权法上的「表达与思想合并」情形，不构成复制证据。上表的共享 n-gram 示例可人工复核。

#### 共享 n-gram 示例（供人工判断是否为通用写法）

| 目标文件 | 共享 n-gram（token 序列） |
| --- | --- |
| `src/scitrace/ports/common.py` | `( default = 0 , ge = 0` |
| `src/scitrace/ports/common.py` | `( extra = "forbid" ) name : str` |
| `src/scitrace/ports/common.py` | `: int = Field ( default = 0` |
| `src/scitrace/ports/fulltext_index.py` | `( self , query : str , k` |
| `src/scitrace/ports/fulltext_index.py` | `, query : str , k : int` |
| `src/scitrace/ports/fulltext_index.py` | `, str ] \| None = None ,` |
| `src/scitrace/domain/evidence.py` | `( cls , value : str ) ->` |
| `src/scitrace/domain/evidence.py` | `) @ field_validator ( "summary" ) @ classmethod` |
| `src/scitrace/domain/evidence.py` | `, value : str ) -> str :` |
| `src/scitrace/domain/fragment.py` | `( cls , value : str ) ->` |
| `src/scitrace/domain/fragment.py` | `, value : str ) -> str :` |
| `src/scitrace/domain/fragment.py` | `: dict [ str , str ] =` |
| `src/scitrace/ports/llm.py` | `( default = 0 , ge = 0` |
| `src/scitrace/ports/llm.py` | `, Any ] = Field ( default_factory =` |
| `src/scitrace/ports/llm.py` | `: bool = Field ( default = True` |

> ℹ️ 另有 26 个文件的 n-gram 总数不足 300，**不参与单文件判定**：小样本下 containment 方差极大（十几个通用 token 序列即可超过 5%），属于统计假象而非复制证据。这些文件仍列在上表中，其最高值为 13.8298%（`src/scitrace/ports/common.py`），请结合共享 n-gram 示例人工核验。

## 2. 逐行精确复制

**状态：** ✅ 通过　　**关键指标：** 逐字相同行 385（实质性 0 / 通用惯用式 385）；最长对齐片段 2 行　　**阈值：** 实质性 ≤ 0 行 且 对齐片段 < 5 行

- 上游可比对代码行索引：**12,237** 条唯一非空行（`.py`，已剔除 <12 字符与通用样板）
- 目标侧扫描的非空行：**17,765** 行
- 命中（逐字相同）：**385** 行，其中 **实质性 0** 行、通用惯用式 385 行
- 属于通用样板（import / `if __name__` 等）而完全忽略：**0** 行
- 行号同步递增的对齐片段：**5** 段，最长 **2** 行

> 说明：`通用惯用式`（类型守卫、异常捕获、单关键字语句、关键字实参行等）与 <12 字符规则同性质，属于“必然重复”，只展示不判失败；真正的复制证据是 **实质性行** 与 **对齐片段**。

#### 前 20 条匹配

| # | 层级 | 目标位置 | 上游位置（首个） | 行内容预览 |
| --- | --- | --- | --- | --- |
| 1 | `通用惯用式` | `benchmarks/ablation.py`:85 | `src/paperqa/clients/journal_quality.py`:216 | `asyncio.run(main())` |
| 2 | `通用惯用式` | `benchmarks/boundary_corpus.py`:367 | `src/paperqa/utils.py`:378 | `return target` |
| 3 | `通用惯用式` | `benchmarks/qa_eval.py`:41 | `packages/paper-qa-nemotron/src/paperqa_nemotron/api.py`:62 | `logger = logging.getLogger(__name__)` |
| 4 | `通用惯用式` | `benchmarks/qa_eval.py`:61 | `src/paperqa/types.py`:331 | `question: str` |
| 5 | `通用惯用式` | `benchmarks/qa_eval.py`:107 | `src/paperqa/agents/env.py`:297 | `return False` |
| 6 | `通用惯用式` | `benchmarks/qa_eval.py`:350 | `src/paperqa/agents/main.py`:116 | `logger.info(` |
| 7 | `通用惯用式` | `benchmarks/qa_eval.py`:402 | `src/paperqa/agents/__init__.py`:176 | `parser.add_argument(` |
| 8 | `通用惯用式` | `benchmarks/qa_eval.py`:404 | `src/paperqa/agents/tools.py`:55 | `default=None,` |
| 9 | `通用惯用式` | `benchmarks/qa_eval.py`:407 | `src/paperqa/agents/__init__.py`:176 | `parser.add_argument(` |
| 10 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:51 | `packages/paper-qa-nemotron/src/paperqa_nemotron/api.py`:62 | `logger = logging.getLogger(__name__)` |
| 11 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:166 | `src/paperqa/llms.py`:71 | `def __len__(self) -> int:` |
| 12 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:169 | `src/paperqa/clients/__init__.py`:78 | `def __repr__(self) -> str:` |
| 13 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:235 | `src/paperqa/agents/main.py`:116 | `logger.info(` |
| 14 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:272 | `src/paperqa/llms.py`:86 | `def clear(self) -> None:` |
| 15 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:325 | `src/paperqa/agents/main.py`:437 | `return results` |
| 16 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:371 | `src/paperqa/llms.py`:132 | `if fetch_k < k:` |
| 17 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:419 | `src/paperqa/core.py`:376 | `score=score,` |
| 18 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:441 | `src/paperqa/utils.py`:529 | `directory.mkdir(parents=True, exist_ok=True)` |
| 19 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:455 | `src/paperqa/agents/main.py`:116 | `logger.info(` |
| 20 | `通用惯用式` | `src/scitrace/adapters/indexes/numpy_vector.py`:483 | `packages/paper-qa-nemotron/src/paperqa_nemotron/reader.py`:217 | `logger.warning(` |

#### 行号同步递增的对齐片段

| # | 长度 | 目标文件 | 目标行区间 | 上游文件 | 上游行区间 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | `src/scitrace/config/settings.py` | 263–264 | `src/paperqa/settings.py` | 132–133 |
| 2 | 2 | `src/scitrace/domain/session.py` | 77–78 | `src/paperqa/settings.py` | 160–161 |
| 3 | 2 | `src/scitrace/domain/session.py` | 228–229 | `src/paperqa/settings.py` | 132–133 |
| 4 | 2 | `src/scitrace/ports/llm.py` | 81–82 | `src/paperqa/settings.py` | 160–161 |
| 5 | 2 | `src/scitrace/util/hashing.py` | 53–54 | `src/paperqa/utils.py` | 118–119 |

## 3. 字符串字面量相似度（阈值 0.60）

**状态：** ✅ 通过　　**关键指标：** 违规相似串 0 条（≥0.60 相似串 41 条）；专有标识符违规 0 处（说明性提及 10 处）　　**阈值：** 相似度 ≥ 0.60 且匹配字符 ≥ 32 的相似串 = 0 且 代码中的标识符违规 = 0

- 上游字符串字面量（≥20 字符，已跳过 docstring）：**1,778** 条
- 目标字符串字面量（≥20 字符，已跳过 docstring）：**762** 条
- 与上游最相似串相似度 ≥ 0.60 的目标串：**41** 条
- 其中**判定为违规**（实际匹配字符数 ≥ 32）：**0** 条
- 说明：相似度高但匹配字符数不足的字符串，多为 DOI / URL / arXiv 编号 / 作者名 / 纯标识符等**事实性短串**（DOI 天生彼此相似），不含受保护表达，因此不判失败；可用 `--min-matched-chars` 调整该门槛。
- 专有标识符：违规 **0** 处、说明性提及（注释 / docstring）**10** 处

#### 相似度最高的前 20 对字符串（目标 ← 上游）

| # | 相似度 | 匹配字符 | 目标位置 | 上游位置 | 目标字符串 | 上游字符串 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000 · | 23 | `src/scitrace/adapters/metadata/__init__.py`:31 | `tests/test_clients.py`:670 | `SemanticScholarProvider` | `SemanticScholarProvider` |
| 2 | 1.000 · | 23 | `src/scitrace/adapters/metadata/semantic_scholar.py`:41 | `tests/test_clients.py`:670 | `SemanticScholarProvider` | `SemanticScholarProvider` |
| 3 | 1.000 · | 22 | `src/scitrace/adapters/metadata/crossref.py`:104 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 4 | 1.000 · | 22 | `tests/test_metadata_providers.py`:193 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 5 | 1.000 · | 22 | `tests/test_metadata_providers.py`:212 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 6 | 1.000 · | 22 | `tests/test_metadata_providers.py`:217 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 7 | 1.000 · | 22 | `tests/test_metadata_providers.py`:222 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 8 | 1.000 · | 22 | `tests/test_metadata_providers.py`:820 | `src/paperqa/clients/crossref.py`:60 | `is-referenced-by-count` | `is-referenced-by-count` |
| 9 | 0.920 · | 23 | `tests/test_optimizations.py`:120 | `tests/test_clients.py`:784 | `Attention Is All You Need` | `Attention is All you Need` |
| 10 | 0.885 · | 23 | `tests/test_optimizations.py`:120 | `tests/test_clients.py`:784 | `"Attention Is All You Need"` | `Attention is All you Need` |
| 11 | 0.841 · | 29 | `tests/test_similarity_audit.py`:266 | `tests/test_paperqa.py`:2740 | `KNOWN = "https://doi.org/10.31224/4087"\n` | `https://doi.org/10.31224/4087` |
| 12 | 0.792 · | 19 | `src/scitrace/config/settings.py`:102 | `tests/test_paperqa.py`:1161 | `deepseek/deepseek-chat` | `deepseek/deepseek-reasoner` |
| 13 | 0.792 · | 19 | `tests/test_config.py`:75 | `tests/test_paperqa.py`:1161 | `deepseek/deepseek-chat` | `deepseek/deepseek-reasoner` |
| 14 | 0.760 · | 19 | `tests/test_ingest.py`:298 | `tests/test_clients.py`:792 | `10.48550/arxiv.2409.13740` | `10.48550/arxiv.1706.03762` |
| 15 | 0.694 · | 17 | `tests/test_agent_runtime.py`:644 | `src/paperqa/clients/client_models.py`:133 | `metadata source down` | `Metadata service is down for ` |
| 16 | 0.683 · | 14 | `tests/test_ingest.py`:79 | `src/paperqa/agents/search.py`:476 | `should not be indexed` | ` could not be found.` |
| 17 | 0.683 · | 14 | `tests/test_ingest.py`:80 | `src/paperqa/agents/search.py`:476 | `should not be indexed` | ` could not be found.` |
| 18 | 0.676 · | 23 | `tests/test_similarity_audit.py`:271 | `tests/test_paperqa.py`:2740 | `SAMPLE = "https://doi.org/10.1234/ABC"\n` | `https://doi.org/10.31224/4087` |
| 19 | 0.667 · | 17 | `tests/test_llm_adapters.py`:266 | `src/paperqa/clients/semantic_scholar.py`:310 | `Invalid API key provided` | `Valid DOI must be provided.` |
| 20 | 0.667 · | 14 | `src/scitrace/domain/evidence.py`:66 | `tests/test_paperqa.py`:1451 | `Evidence.summary 不能为空` | `evidence_skip_summary` |

_标记说明：❗ = 违规（相似度与匹配字符数同时达标）；· = 相似度达标但匹配字符数不足，不计为违规。_

#### 上游专有标识符专项检查

扫描目标 `.py` / 配置文件中是否出现 `pqac`、`paperqa`、`paper-qa`、`paper_qa`、`pqa`（大小写不敏感子串匹配；审计工具自身文件已排除）。出现在注释或 docstring 中的**归属声明 / 说明性提及**记为警告，出现在代码或普通字符串中的记为违规。

**违规明细（代码 / 普通字符串）**

_无。_

**说明性提及（注释 / docstring，不判失败）**

| # | 文件 | 行号 | 命中 | 位置性质 | 行内容 |
| --- | --- | --- | --- | --- | --- |
| 1 | `pyproject.toml` | 33 | `paper-qa` | `comment` | `# 明确不依赖任何 paper-qa* / fhaviary / aviary / fhlmi / ldp ——` |
| 2 | `src/scitrace/__init__.py` | 3 | `paperqa` | `docstring` | `设计范式参考 PaperQA2（arXiv:2409.13740）所述的公开方法，代码为独立实现。` |
| 3 | `src/scitrace/adapters/parsers/pypdf_parser.py` | 146 | `paperqa` | `comment` | ```# 内嵌标题是 "paperqa2"，而 ``apply_patch`` 又规定"不覆盖已有值"，``` |
| 4 | `src/scitrace/adapters/parsers/pypdf_parser.py` | 148 | `paperqa` | `comment` | ```# ``(anonndpaperqa2 pages 2-3)``。``` |
| 5 | `src/scitrace/pipeline/chunking.py` | 341 | `paperqa` | `docstring` | `**真实的论文上不成立**：实测 PaperQA2 原文（25 页）的正文只有 9 页，` |
| 6 | `src/scitrace/pipeline/chunking.py` | 342 | `paperqa` | `docstring` | `第 9 页末尾是 References，而第 12–25 页是 "8 Methods / 8.1 PaperQA` |
| 7 | `src/scitrace/pipeline/ingest.py` | 327 | `paperqa` | `comment` | ```# 文内引用退化成 ``(anonndpaperqa2 pages 2-3)``。``` |
| 8 | `src/scitrace/pipeline/ingest.py` | 357 | `paperqa` | `comment` | ```# （``01_Lewis2020_RAG奠基论文`` 尚可，``paperqa2`` 则完全无用）。``` |
| 9 | `tests/test_chunking.py` | 165 | `paperqa` | `docstring` | `回归自真实数据：PaperQA2 原文正文只有 9 页，第 9 页末尾是 References，` |
| 10 | `tests/test_ingest.py` | 287 | `paperqa` | `docstring` | ```补不上，文内引用退化成 ``(anonndpaperqa2 pages 2-3)``；``` |

_上述提及属于归属声明或说明性文字（例如指明设计范式来源、声明不依赖上游包），不含受保护的表达，因此不计为违规；如需最严格口径可加 `--strict-identifiers`。_

## 4. 依赖审计

**状态：** ✅ 通过　　**关键指标：** 禁用依赖 0 处；警告 0 处　　**阈值：** 禁用依赖 = 0

- 禁用清单（命中即失败）：`paperqa`、`fhaviary`、`aviary`、`fhlmi`、`ldp`
- 仅警告：`lmi`
- 已扫描的清单文件：`pyproject.toml`
- 审计工具自身文件（文件名含 `similarity_audit`、`tools/README.md`）已排除，避免自我指控。

#### 违规明细

_未发现禁用依赖。_

#### 警告明细（不判失败）

_无警告。_

## 5. 资产 / 文档审计（文档 n=12）

**状态：** ✅ 通过　　**关键指标：** 哈希相同资产 0 个；文档整体 containment 0.0400%（最高 0.6594%）　　**阈值：** 哈希相同 = 0 且 文档整体 ≤ 2.00% 且 单文档 ≤ 5.00%

#### 5.1 资产内容哈希（SHA-256）

- 上游可比对 blob：**163** 个（已跳过 0 字节文件）
- 目标文件哈希相同者：**0** 个

_未发现与上游 git 跟踪文件内容完全相同的资产。_

#### 5.2 文档 token n-gram（n=12）

- 上游文本文件参与比对：**162** 个，n-gram 池 **1,558,270** 个
- 目标 `.md`/`.rst` 文件：**19** 个，n-gram 池 **57,566** 个
- 整体 containment **0.0400%**，Jaccard **0.0014%**

| # | 目标文档 | containment | 共有 n-gram | 文档 n-gram 总数 |
| --- | --- | --- | --- | --- |
| 1 | `README.md` | 0.6594% | 14 | 2,123 |
| 2 | `docs/eval_heldout27_agentic_k10.md` | 0.5571% | 4 | 718 |
| 3 | `docs/eval_heldout27_agentic_k3.md` | 0.5556% | 4 | 720 |
| 4 | `docs/eval_heldout27_agentic_k5.md` | 0.5556% | 4 | 720 |
| 5 | `docs/eval_heldout27_det_k5.md` | 0.5540% | 4 | 722 |
| 6 | `docs/eval_heldout27_det_k10.md` | 0.5533% | 4 | 723 |
| 7 | `docs/eval_deterministic_9q.md` | 0.4535% | 4 | 882 |
| 8 | `docs/eval_agentic_9q_k5.md` | 0.4386% | 4 | 912 |
| 9 | `docs/eval_agentic_9q_k3.md` | 0.4367% | 4 | 916 |
| 10 | `docs/eval_agentic_9q_crossencoder.md` | 0.4338% | 4 | 922 |
| 11 | `docs/eval_agentic_9q.md` | 0.4320% | 4 | 926 |
| 12 | `benchmarks/data/eval36_report.md` | 0.4283% | 4 | 934 |
| 13 | `docs/eval_36q_deterministic.md` | 0.4278% | 4 | 935 |
| 14 | `tools/README.md` | 0.2564% | 2 | 780 |
| 15 | `docs/SPEC.md` | 0.0957% | 9 | 9,403 |
| 16 | `docs/重构方案.md` | 0.0843% | 5 | 5,934 |
| 17 | `docs/PROVENANCE.md` | 0.0669% | 5 | 7,478 |
| 18 | `docs/COMPARISON.md` | 0.0648% | 4 | 6,170 |
| 19 | `docs/EXPERIMENTS.md` | 0.0359% | 7 | 19,481 |

---

## 如何解读本报告

### 本报告能够证明什么

- 在给定参数（n-gram 长度、阈值）下，目标项目中**不存在**与上游 HEAD 快照逐字相同的
  源码行、字符串字面量或 token n-gram 片段；
- 目标项目**没有**依赖上游发行包及其同组织配套包（`paperqa` / `fhaviary` / `aviary` / `fhlmi` / `ldp`）；
- 目标项目**没有**搬运上游仓库中被 git 跟踪的资产文件（内容哈希一致者）。

以上都属于"**没有复制表达**"的证据。

### 本报告不能证明什么

- **不能证明"思想独立"**。n-gram 与字符串阈值只能捕捉"表达"层面的复制：算法思路、
  模块划分、命名风格、Prompt 的语义设计、调用链结构仍可能受到上游启发而低于阈值；
  改写（paraphrase）、翻译、token 重排、变量重命名等手法都可能在阈值之下不被发现。
- **不能替代法律意见**。Apache-2.0 允许在有归属声明的前提下复用；本审计的目标是支撑
  "代码为独立编写"的声明，而不是判定许可合规性。
- **阈值是工程判据，不是"独创性"的度量**。低于阈值 ≠ 独创，高于阈值 ≠ 侵权。

### 思想层面的原创性依赖洁净室流程

本脚本只能审计"文本"，无法审计"过程"。思想层面的原创性证据应当来自
`docs/PROVENANCE.md` 中记录的洁净室流程，例如：需求是如何被**转述**为规格说明的、
谁在什么时间接触过上游代码、实现者是否在未接触上游源码的情况下完成设计、
上游代码是否仅被用于事后比对（正如本脚本所做的）。建议将本报告与 `PROVENANCE.md`
一并作为原创性声明的附件。


## 方法与排除项

- **参考侧只读 HEAD**：文件列表由 `git -C <reference> ls-files` 得到，内容由 `git -C <reference> show HEAD:<path>` 读取；工作区中的未跟踪文件与本地改动（例如上游仓库里用户自己新增的笔记、`my_papers/`）一律不参与比对。
- **目标侧遍历排除**：`.git/`、`__pycache__/`、`.venv/`、`node_modules/`、`build/`、`dist/`、`*.egg-info/`、审计报告自身，以及 `--exclude` 指定的模式。
- **分词**：`.py` 使用保留注释与字符串字面量的正则分词器；`.md`/散文使用词/字级分词器。
- **行级比对**：仅比较 `.py`；忽略去空白后长度 < 12 字符的行。命中行分三档：`样板`（import / `if __name__` 等，完全忽略）、`通用惯用式`（类型守卫、异常捕获、字段声明、关键字实参行、调用头等，计数展示但不判失败）、`实质性`（判失败）。此外还检测**行号同步递增的对齐片段**：若目标第 t 行 = 上游第 r 行且 t+1 = r+1……连续出现，即使每行都很短也视为整块搬运。
- **字符串比对**：仅比较长度 ≥ 20 字符、且非 docstring 的字面量；docstring 已由检查 1 覆盖，避免重复计数。判违规需同时满足「相似度 ≥ 0.60」与「实际匹配字符数 ≥ 32」，后者用于排除 DOI / URL / 作者名 / 纯标识符等天然相似的事实性短串。
- **专有标识符**：出现在注释 / docstring 中的归属声明或说明性提及记为警告，出现在代码或普通字符串中的记为违规（`--strict-identifiers` 可切换为全部判失败）。
- **依赖审计**：审计工具自身文件（文件名含 `similarity_audit` 者、`tools/README.md`）会被排除，因为它们必然包含禁用词表。
- **资产哈希**：跳过 0 字节文件（空文件哈希相同不具信息量）。
- **未经检验的维度**：语义相似度、算法/架构相似度、Prompt 的改写式抄袭、由上游代码转译（例如 Python→其它语言）后的抄袭，均不在本脚本能力范围内。

## 复现命令

```bash
python tools/similarity_audit.py --target /home/wby/MyPaperQA --reference /home/wby/PaperQA --report docs/AUDIT.md --ngram 8 --fail-threshold 0.02
```

"""摄入编排：把语料目录变成可检索的索引。

对应 SPEC §3.1 的增量语义。本模块是"文件系统"与"索引"之间的粘合层，
也是**唯一**知道 manifest 存在的地方。

## 为什么 manifest 是一等公民

没有它，"重新索引"只能全量重做——一本 300 页的 PDF 解析 + 切成百上千个片段 +
逐个向量化，成本以分钟计；而用户的实际操作模式是"往目录里丢两篇新论文，
再跑一次 index"。manifest 把这件事变成"只处理变化的部分"。

它的键是**相对路径**而非内容哈希：这样文件改名会被识别为
"删除旧的 + 新增新的"，而不是"两个不同的文档"——这与用户的直觉一致。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import anyio
from pydantic import BaseModel, ConfigDict, Field

from scitrace import SCHEMA_VERSION
from scitrace.config import ChunkingSettings
from scitrace.domain import Fragment, Source, SourcePatch, make_source_key
from scitrace.domain.session import Usage
from scitrace.pipeline.chunking import chunk_document
from scitrace.pipeline.title_inference import LLMTitleInferrer
from scitrace.ports import (
    DocumentParser,
    EmbeddingClient,
    FullTextIndex,
    MetadataResolver,
    ParseError,
    VectorIndex,
)
from scitrace.service import SourceStore
from scitrace.util.hashing import hash_file
from scitrace.util.sanitize import SanitizedModel, sanitize_unicode
from scitrace.util.text import find_arxiv_id, find_doi

logger = logging.getLogger(__name__)

__all__ = [
    "IngestPipeline",
    "IngestReport",
    "Manifest",
    "ManifestEntry",
    "ManifestStore",
]

MANIFEST_FILENAME = "manifest.json"

#: 并发解析的文件数。解析是 CPU 密集型且经线程池执行，适度并发能显著缩短
#: 首次建索引的时间；过高则会让内存与磁盘 IO 成为瓶颈反而变慢。
DEFAULT_PARSE_CONCURRENCY = 4

#: 视为可摄入的扩展名（PDF 由解析器自行声明支持，这里列出的是目录扫描时的白名单）。
_ALWAYS_SCAN_SUFFIXES = frozenset({".pdf", ".txt", ".md", ".markdown", ".text"})


class ManifestEntry(SanitizedModel):
    """清单中的单条记录。"""

    hash: str
    source_key: str
    status: Literal["ok", "failed", "skipped"] = "ok"
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    fragment_count: int = Field(default=0, ge=0)


class Manifest(SanitizedModel):
    """索引清单：``相对路径 -> 记录``。

    ``schema`` 参与版本判断：schema 升级后旧清单会被整体丢弃并全量重建，
    因为字段语义可能已经变了，沿用旧清单比重建更危险。
    """

    schema_version: int = SCHEMA_VERSION
    entries: dict[str, ManifestEntry] = Field(default_factory=dict)

    def is_compatible(self) -> bool:
        """清单的 schema 是否与当前版本一致。"""
        return self.schema_version == SCHEMA_VERSION


class ManifestStore:
    """清单的读写。

    刻意不做原子写之外的任何事：清单的正确性依赖调用方在**全部**处理完成后
    一次性保存，而不是边处理边写——中途崩溃留下的半成品清单会让下次索引
    误以为某些文件已经处理过。
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.path = self.index_dir / MANIFEST_FILENAME

    def load(self) -> Manifest:
        """读取清单。文件不存在、损坏或 schema 不兼容时返回空清单。"""
        if not self.path.is_file():
            return Manifest()
        try:
            manifest = Manifest.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 — 损坏的清单应当降级为全量重建
            logger.warning("清单无法解析，将全量重建：%s", error)
            return Manifest()
        if not manifest.is_compatible():
            logger.warning(
                "清单 schema 版本不兼容（%d ≠ %d），将全量重建",
                manifest.schema_version,
                SCHEMA_VERSION,
            )
            return Manifest()
        return manifest

    def save(self, manifest: Manifest) -> None:
        """原子写入清单。

        先写临时文件再 ``replace``：直接覆写在进程被杀时会留下截断的 JSON，
        下次索引就会因为解析失败而全量重建——一次意外停电的代价被放大成
        一次完整重建。
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        # 与 SourceStore 同理的最后一道兜底：清洗必须发生在序列化**之前**，
        # 因为孤立代理项会让 model_dump_json() 直接抛错。
        temporary.write_text(
            sanitize_unicode(manifest.model_dump_json(indent=2)), encoding="utf-8"
        )
        temporary.replace(self.path)


@dataclass
class IngestReport:
    """一次摄入的结果。"""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    sources: dict[str, Source] = field(default_factory=dict)
    fragment_count: int = 0
    duplicate_sources: dict[str, list[str]] = field(default_factory=dict)
    #: 摄入期间产生的 LLM 用量。目前只有标题推断会用到——
    #: 把它记下来，才能回答"开着它索引 500 篇论文要多少钱"这个决定性问题。
    usage: Usage = field(default_factory=Usage)

    @property
    def processed(self) -> int:
        """实际发生了解析与索引的文件数。"""
        return len(self.added) + len(self.updated)

    @property
    def ok(self) -> bool:
        """是否没有失败项。"""
        return not self.failed

    def summary(self) -> str:
        """生成给 CLI 显示的摘要行。"""
        parts = [
            f"新增 {len(self.added)}",
            f"更新 {len(self.updated)}",
            f"跳过 {len(self.skipped)}",
            f"删除 {len(self.removed)}",
            f"失败 {len(self.failed)}",
            f"片段 {self.fragment_count}",
        ]
        return "，".join(parts)


def _split_author_string(raw: str | None) -> list[str] | None:
    """把 PDF 元数据里的作者串拆成列表。

    实际遇到的写法至少有四种：``"Smith, John; Doe, Jane"``、
    ``"John Smith and Jane Doe"``、``"John Smith, Jane Doe"``、以及换行分隔。
    这里按优先级依次尝试分隔符，尽量不把"Smith, John"这种"姓, 名"形式拆错：
    先用分号/换行这类**无歧义**分隔符；只有当串里不含分号、且逗号数量为
    偶数时才用逗号分隔（"A, B, C, D" 更可能是"名 姓, 名 姓"）。
    """
    if not raw or not raw.strip():
        return None
    text = raw.replace("\r", "\n").strip()
    for separator in (";", "\n", " and ", " & "):
        if separator in text:
            parts = [item.strip() for item in text.split(separator)]
            return [item for item in parts if item] or None
    if "," in text:
        segments = [item.strip() for item in text.split(",") if item.strip()]
        if len(segments) % 2 == 0:
            return [f"{segments[index + 1]} {segments[index]}" for index in range(0, len(segments), 2)]
        return segments
    return [text]


class IngestPipeline:
    """把语料目录摄入索引。

    本类**不负责**创建索引与嵌入器，只负责编排：由装配层（``scitrace.factory``）
    决定"用哪个向量库、哪个嵌入模型"，并在此注入。因此测试可以注入内存实现，
    无需联网或加载模型。
    """

    def __init__(
        self,
        *,
        index_dir: Path,
        chunking: ChunkingSettings | None = None,
        parser_resolver: Callable[[Path], DocumentParser] | None = None,
        vector_index: VectorIndex | None = None,
        fulltext_index: FullTextIndex | None = None,
        embedder: EmbeddingClient | None = None,
        resolver: MetadataResolver | None = None,
        title_inferrer: LLMTitleInferrer | None = None,
        max_file_mb: float = 200.0,
        parse_concurrency: int = DEFAULT_PARSE_CONCURRENCY,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.chunking = chunking or ChunkingSettings()
        self.parser_resolver = parser_resolver
        self.vector_index = vector_index
        self.fulltext_index = fulltext_index
        self.embedder = embedder
        self.resolver = resolver
        self.title_inferrer = title_inferrer
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)
        self.parse_concurrency = max(1, parse_concurrency)
        self.manifest_store = ManifestStore(self.index_dir)
        # 文献元数据必须与索引一同持久化：查询阶段要靠它把 source_key 渲染成
        # 可读引用与 BibTeX，而索引里只存 fragment 与向量。
        self.source_store = SourceStore(self.index_dir)
        #: 摄入结束后，索引里的**全部**文献元数据。
        #:
        #: 调用方（尤其是"同一进程内先摄入再提问"的场景）需要它来刷新自己持有的
        #: 元数据视图——否则会拿着构造时的空快照去渲染引用，表现为引用全是
        #: ``unknown``，而索引里其实有完整信息。放在这里而不是让调用方各自去
        #: 读盘，是因为**状态的改变发生在这里**，刷新责任就该在这里收口。
        self.sources: dict[str, Source] = {}

    # ------------------------------------------------------------------ 发现 --

    def discover(self, roots: Sequence[Path]) -> dict[str, Path]:
        """扫描语料根目录，返回 ``标识 -> 绝对路径``。

        标识形如 ``<根目录名>/<相对路径>``：带根目录名是为了在同时索引多个语料时
        避免同名文件互相覆盖，也让报告里的路径一眼能看出文件属于哪个语料。

        跳过隐藏目录与隐藏文件（``.git``、``.DS_Store`` 之类），
        以及超过大小上限的文件（解析一本 2 GB 的扫描书会耗尽内存，
        而且它本来也不会有文本层）。
        """
        found: dict[str, Path] = {}
        for root in roots:
            root_path = Path(root).expanduser().resolve()
            if not root_path.exists():
                logger.warning("语料路径不存在，已跳过：%s", root_path)
                continue
            if root_path.is_file():
                found[self._identifier(root_path.parent, root_path)] = root_path
                continue
            for path in sorted(root_path.rglob("*")):
                if not path.is_file():
                    continue
                if any(part.startswith(".") for part in path.relative_to(root_path).parts):
                    continue
                if path.suffix.lower() not in _ALWAYS_SCAN_SUFFIXES:
                    continue
                if path.stat().st_size > self.max_file_bytes:
                    logger.warning("文件超过大小上限（%.0f MB），已跳过：%s", self.max_file_bytes / 1048576, path)
                    continue
                found[self._identifier(root_path, path)] = path
        return found

    @staticmethod
    def _identifier(root: Path, path: Path) -> str:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - 理论上不会发生
            relative = path.name
        return f"{root.name}/{relative}"

    # -------------------------------------------------------------- 单文件 --

    def _resolve_parser(self, path: Path) -> DocumentParser:
        """为该文件解析出一个解析器。

        **解析器由外部注入**，本模块不 import 任何适配器——这是"pipeline 只依赖
        domain/ports/util"这条架构约束的一部分（``tests/test_layering.py`` 会机械校验）。
        注入点由装配层提供，因此"用哪个解析后端"仍然只在一处决定。
        """
        if self.parser_resolver is None:
            raise ValueError(
                f"未注入解析器解析函数，无法处理 {path.name}。"
                "请在装配层（scitrace.factory）传入 parser_resolver。"
            )
        return self.parser_resolver(path)

    async def ingest_file(self, path: Path, *, identifier: str, digest: str) -> tuple[Source, list[Fragment]]:
        """解析、补全元数据、分块，返回文献与片段。

        Args:
            path: 文件绝对路径。
            identifier: 清单中的标识（``<根目录名>/<相对路径>``）。
            digest: 文件内容哈希（调用方已算好，避免重复读取整个文件）。

        Raises:
            ParseError: 解析失败。由调用方捕获并记为 ``status=failed``。
            ValueError: 没有解析器能处理该文件。
        """
        document = await self._resolve_parser(path).parse(path)

        hints = document.hints
        first_page = (document.pages[0].text if document.pages else "")[:4000]
        doi = find_doi(first_page)
        if doi is None:
            # 预印本常只印 ``arXiv:2409.13740`` 而不印 DOI。arXiv 为每篇预印本
            # 分配了固定格式的 DOI（``10.48550/arXiv.<编号>``），据此构出 DOI
            # 就能让元数据来源走**精确端点**而不是模糊检索，命中率与准确率
            # 都是另一个量级。实测：不加这一步时，一篇 arXiv 论文的标题补不上，
            # 文内引用退化成 ``(anonndpaperqa2 pages 2-3)``。
            arxiv_id = find_arxiv_id(first_page)
            if arxiv_id:
                doi = f"10.48550/arXiv.{arxiv_id}"
                logger.debug("由 arXiv 编号构造 DOI：%s", doi)
        # 本地没有可用标题时，用 LLM 从首页推断一个——它既是给元数据来源的**查询线索**
        # （没有标题与 DOI 时，来源根本无从查起，形成"没标题→查不到→永远没标题"的死循环），
        # 也是最终的兜底标题。默认关闭，因为它把索引从纯本地计算变成依赖外部服务。
        inferred_title: str | None = None
        if self.title_inferrer is not None and not hints.get("title"):
            inferred_title = await self.title_inferrer.infer(
                document.pages[0].text if document.pages else ""
            )

        patch = SourcePatch(
            # 只有**真实的内嵌标题**才作为查询线索；文件名不在其中——
            # 用文件名去 Crossref 模糊检索只会搜出无关论文。
            title=hints.get("title") or inferred_title,
            authors=_split_author_string(hints.get("authors")),
            doi=doi,
        )
        if self.resolver is not None:
            patch = await self.resolver.enrich(patch)

        source = Source(
            key=make_source_key(doi=patch.doi, content_hash=digest),
            content_hash=digest,
            rel_path=identifier,
            # 优先级：内嵌标题 > LLM 推断 > 元数据来源补的 > 文件名。
            # LLM 推断排在文件名之前，因为文件名对读者几乎没有信息量
            # （``01_Lewis2020_RAG奠基论文`` 尚可，``paperqa2`` 则完全无用）。
            title=patch.title or inferred_title or hints.get("fallback_title") or path.stem,
            authors=patch.authors or [],
            year=patch.year,
            doi=patch.doi,
            venue=patch.venue,
            abstract=patch.abstract,
            citation_count=patch.citation_count,
            is_oa=patch.is_oa,
            oa_url=patch.oa_url,
            quality_tier=patch.quality_tier or 0,
            retracted=patch.retracted is True,
            metadata_sources=list(self.resolver.name.split("+")) if self.resolver else [],
        )

        fragments = chunk_document(
            document, source_key=source.key, settings=self.chunking, document_hash=digest
        )
        return source, fragments

    async def _index_fragments(self, source: Source, fragments: Sequence[Fragment]) -> None:
        """把片段写入双索引。"""
        if not fragments:
            return
        if self.fulltext_index is not None:
            await self.fulltext_index.add(fragments, titles={source.key: source.title})
        if self.vector_index is not None and self.embedder is not None:
            vectors = await self.embedder.embed([item.text for item in fragments], kind="document")
            if len(vectors) != len(fragments):
                raise RuntimeError(
                    f"嵌入结果数量与片段不匹配：{len(vectors)} ≠ {len(fragments)}"
                )
            await self.vector_index.add(list(zip(fragments, vectors, strict=True)))

    async def _unindex_source(self, source_key: str) -> None:
        """从双索引移除某篇文献的全部片段。"""
        if self.vector_index is not None:
            await self.vector_index.remove(source_key)
        if self.fulltext_index is not None:
            await self.fulltext_index.remove(source_key)

    # ---------------------------------------------------------------- 编排 --

    async def run(self, roots: Sequence[Path], *, rebuild: bool = False) -> IngestReport:
        """执行一次摄入。

        Args:
            roots: 语料根目录（或单个文件）。
            rebuild: 忽略既有清单，全量重建。

        Returns:
            摄入报告。即使存在失败文件也会正常返回（``report.ok`` 为 ``False``），
            由调用方决定退出码——单个坏文件不应中断整批。
        """
        report = IngestReport()
        files = self.discover(roots)
        manifest = Manifest() if rebuild else self.manifest_store.load()
        stored_sources = {} if rebuild else self.source_store.load_sources()
        if rebuild:
            logger.info("全量重建：忽略既有清单与元数据")
            await self._clear_indexes()

        pending: list[tuple[str, Path, str]] = []
        for identifier, path in sorted(files.items()):
            try:
                digest = hash_file(path)
            except OSError as error:
                report.failed.append((identifier, f"无法读取：{error}"))
                continue
            entry = manifest.entries.get(identifier)
            # 失败过的文件即使哈希未变也要重试：用户可能刚刚补装了依赖或修好了文件。
            if entry is not None and entry.hash == digest and entry.status == "ok":
                report.skipped.append(identifier)
                continue
            pending.append((identifier, path, digest))

        await self._process_pending(pending, manifest, report, stored_sources)

        # 磁盘上已不存在的文件 → 从索引、清单与元数据中一并移除
        for identifier in [key for key in manifest.entries if key not in files]:
            entry = manifest.entries.pop(identifier)
            if entry.status == "ok":
                await self._unindex_source(entry.source_key)
            stored_sources.pop(entry.source_key, None)
            report.removed.append(identifier)

        if self.title_inferrer is not None:
            report.usage = report.usage.merge(self.title_inferrer.last_usage)
        report.fragment_count = sum(
            entry.fragment_count for entry in manifest.entries.values() if entry.status == "ok"
        )
        if self.fulltext_index is not None:
            await self.fulltext_index.commit()
        await self._persist_indexes()
        self.manifest_store.save(manifest)
        self.source_store.save_sources(stored_sources)
        self.sources = dict(stored_sources)
        logger.info("摄入完成：%s", report.summary())
        return report

    async def _process_pending(
        self,
        pending: Sequence[tuple[str, Path, str]],
        manifest: Manifest,
        report: IngestReport,
        stored_sources: dict[str, Source],
    ) -> None:
        """并发处理待摄入文件，并把结果写回清单、元数据存储与报告。"""
        if not pending:
            return
        limiter = anyio.Semaphore(self.parse_concurrency)
        lock = anyio.Lock()

        async def record_failure(
            identifier: str, digest: str, reason: str
        ) -> None:
            """把一个文件的失败写进报告与清单。"""
            previous = manifest.entries.get(identifier)
            async with lock:
                report.failed.append((identifier, reason))
                manifest.entries[identifier] = ManifestEntry(
                    hash=digest,
                    source_key=previous.source_key if previous else "",
                    status="failed",
                    error=reason,
                )

        async def process(identifier: str, path: Path, digest: str) -> None:
            """处理单个文件（**不含**异常兜底，由 worker 负责）。"""
            previous = manifest.entries.get(identifier)
            try:
                source, fragments = await self.ingest_file(
                    path, identifier=identifier, digest=digest
                )
            except (ParseError, ValueError) as error:
                logger.warning("摄入失败：%s（%s）", identifier, error)
                await record_failure(identifier, digest, str(error))
                return
            except Exception as error:  # noqa: BLE001 — 未预期错误也要记录而非中断整批
                logger.exception("摄入出现未预期错误：%s", identifier)
                await record_failure(identifier, digest, f"未预期错误：{error}")
                return

            # 先清掉该文件的上一版片段，再写入新片段。
            #
            # 必须无条件清理（而不只是 source_key 变化时）：`fragment_id` 含内容哈希，
            # 内容一变旧 id 就不再被新片段覆盖，不清理会留下陈旧内容——
            # 表现是"我明明改了论文，检索到的还是旧段落"。
            #
            # 唯一的例外是**另一个清单项共用同一个 source_key**（同一 DOI 的
            # 预印本与正式版）。此时 `remove(source_key)` 会连对方的片段一起删掉，
            # 所以跳过清理；代价是这一项自己的旧片段可能残留。
            # 这是已知且刻意接受的取舍——索引接口以 source_key 为删除单位，
            # 要精确到文件级需要端口支持按 fragment_id 删除。
            if previous is not None and previous.source_key:
                shared_with_other_file = any(
                    entry.source_key == previous.source_key and key != identifier
                    for key, entry in manifest.entries.items()
                )
                if not shared_with_other_file:
                    await self._unindex_source(previous.source_key)
                else:
                    logger.debug(
                        "%s 的 source_key 与其他文件共用（%s），跳过清理以避免误删",
                        identifier,
                        previous.source_key,
                    )

            await self._index_fragments(source, fragments)

            async with lock:
                manifest.entries[identifier] = ManifestEntry(
                    hash=digest,
                    source_key=source.key,
                    status="ok",
                    fragment_count=len(fragments),
                )
                if previous is not None:
                    report.updated.append(identifier)
                else:
                    report.added.append(identifier)

                existing = stored_sources.get(source.key)
                if existing is not None and existing.rel_path != source.rel_path:
                    # 同一个 DOI 出现在多个文件里（预印本 + 正式版）：
                    # 归并为同一篇文献是正确的，但要让用户知道发生了什么，
                    # 否则"我明明索引了 5 篇，怎么只有 4 篇"会无从解释。
                    report.duplicate_sources.setdefault(
                        source.key, [existing.rel_path]
                    ).append(source.rel_path)
                stored_sources[source.key] = source
                report.sources[source.key] = source

        async def worker(identifier: str, path: Path, digest: str) -> None:
            """任务组的工作单元：**绝不允许异常逃离本函数**。

            ## 为什么包住整个函数体，而不是逐个调用点

            anyio 的 task group 语义是：**任何**一个子任务的未捕获异常都会
            取消所有兄弟任务。因此每一个裸 ``await`` 都是一次"整批作废"的机会。

            这个坑本项目真实踩过：PDF 解析出的孤立代理项让 tantivy 编码失败，
            异常冲出 worker 掀翻整个 task group，**其余 35 篇已解析完的论文全部作废**。
            当时的修法是给 ``_index_fragments`` 单独加 try——但那只把下一次事故
            推迟到下一个未加保护的语句（事后核查发现 ``_unindex_source`` 就仍在保护之外）。

            正确的位置是**函数边界**：无论哪一步抛出，都只影响这一个文件。
            清单与元数据落盘失败是刻意的例外，由 ``run()`` 统一处理——
            清单与索引不一致比整次失败更危险。
            """
            try:
                async with limiter:
                    await process(identifier, path, digest)
            except Exception as error:  # noqa: BLE001 - 见 docstring
                logger.exception("摄入 %s 时出现未捕获异常", identifier)
                await record_failure(identifier, digest, f"未捕获异常：{error}")

        async with anyio.create_task_group() as task_group:
            for identifier, path, digest in pending:
                task_group.start_soon(worker, identifier, path, digest)

        report.added.sort()
        report.updated.sort()
        report.skipped.sort()
        report.removed.sort()
        report.failed.sort()

    # ------------------------------------------------------------ 索引持久化 --

    async def _clear_indexes(self) -> None:
        if self.vector_index is not None:
            self.vector_index.clear()
        if self.fulltext_index is not None:
            self.fulltext_index.clear()

    async def _persist_indexes(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_index is not None:
            await self.vector_index.persist()
        if self.fulltext_index is not None:
            await self.fulltext_index.commit()

    async def aclose(self) -> None:
        """释放元数据解析器等外部资源。"""
        if self.resolver is not None:
            await self.resolver.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

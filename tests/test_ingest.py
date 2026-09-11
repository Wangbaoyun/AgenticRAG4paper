"""测试摄入编排与增量语义。

本文件覆盖 SPEC §8 的行为契约 **C5（索引重建触发）** 与 **C8（降级路径）**：
摄入的正确性几乎全部体现在"第二次运行时发生了什么"——哪些跳过、哪些重建、
删掉的文件有没有从索引里清干净。这些性质如果错了，症状是"检索里混进了
已经不存在的文档"或"改了文件但检索结果没变"，都属于极难排查的类型。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeEmbedder, FakeFullTextIndex, FakeResolver, FakeVectorIndex
from pdf_fixture import write_text_pdf
from scitrace import SCHEMA_VERSION
from scitrace.domain import SourcePatch
from scitrace.adapters.parsers import select_parser
from scitrace.pipeline.ingest import (
    IngestPipeline,
    IngestReport,
    Manifest,
    ManifestEntry,
    ManifestStore,
)

RESOLVED = SourcePatch(year=2024, venue="Journal of Example Studies")


class Harness:
    """把一次测试所需的管线与替身打包，便于断言。"""

    def __init__(self, root: Path, **kwargs) -> None:
        self.root = root
        self.corpus = root / "corpus"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.index_dir = root / "index"
        self.vector = FakeVectorIndex()
        self.fulltext = FakeFullTextIndex()
        self.embedder = FakeEmbedder()
        self.resolver = FakeResolver()
        self.pipeline = IngestPipeline(
            index_dir=self.index_dir,
            parser_resolver=select_parser,
            vector_index=self.vector,
            fulltext_index=self.fulltext,
            embedder=self.embedder,
            resolver=self.resolver,
            **kwargs,
        )

    def write(self, name: str, content: str) -> Path:
        path = self.corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    async def run(self, *, rebuild: bool = False) -> IngestReport:
        return await self.pipeline.run([self.corpus], rebuild=rebuild)

    def indexed_texts(self) -> str:
        return "\n".join(item.text for item in self.vector.fragments())


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


class TestDiscovery:
    async def test_finds_supported_files_recursively(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough to chunk.")
        harness.write("nested/b.md", "# Beta\n\nBeta content long enough.")
        report = await harness.run()
        assert sorted(report.added) == ["corpus/a.txt", "corpus/nested/b.md"]

    async def test_skips_hidden_files_and_directories(self, harness: Harness) -> None:
        harness.write(".hidden.txt", "should not be indexed")
        harness.write(".git/config.txt", "should not be indexed")
        harness.write("visible.txt", "should be indexed")
        report = await harness.run()
        assert report.added == ["corpus/visible.txt"]

    async def test_skips_unsupported_suffix(self, harness: Harness) -> None:
        harness.write("data.csv", "a,b,c")
        report = await harness.run()
        assert report.added == []

    async def test_skips_oversize_file(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, max_file_mb=0.0001)  # 约 105 字节
        harness.write("small.txt", "tiny")
        harness.write("big.txt", "x" * 5000)
        report = await harness.run()
        assert report.added == ["corpus/small.txt"]

    async def test_missing_root_is_ignored(self, harness: Harness) -> None:
        report = await harness.pipeline.run([harness.root / "does-not-exist"])
        assert report.added == []
        assert report.ok


class TestIncrementalSemantics:
    async def test_first_run_indexes_everything(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough to be chunked properly.")
        harness.write("b.txt", "Beta content long enough to be chunked properly.")

        report = await harness.run()
        assert len(report.added) == 2
        assert report.processed == 2
        assert report.fragment_count == len(harness.vector)
        assert len(harness.fulltext) == len(harness.vector)

    async def test_second_run_skips_unchanged(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough to be chunked properly.")
        await harness.run()
        indexed_before = harness.indexed_texts()
        resolver_calls_before = len(harness.resolver.calls)

        report = await harness.run()
        assert report.skipped == ["corpus/a.txt"]
        assert report.processed == 0
        assert harness.indexed_texts() == indexed_before
        # 跳过必须是真的跳过：不能再调用解析器与元数据源
        assert len(harness.resolver.calls) == resolver_calls_before

    async def test_modified_file_is_reindexed(self, harness: Harness) -> None:
        path = harness.write("a.txt", "Original alpha content for chunking.")
        await harness.run()
        assert "Original alpha" in harness.indexed_texts()

        path.write_text("Replaced alpha content for chunking.", encoding="utf-8")
        report = await harness.run()

        assert report.updated == ["corpus/a.txt"]
        assert "Replaced alpha" in harness.indexed_texts()
        assert "Original alpha" not in harness.indexed_texts(), "旧内容必须被清除"

    async def test_deleted_file_is_purged_everywhere(self, harness: Harness) -> None:
        keep = harness.write("keep.txt", "Keep this content for chunking please.")
        doomed = harness.write("doomed.txt", "Doomed content that will be removed.")
        await harness.run()
        assert "Doomed content" in harness.indexed_texts()

        doomed.unlink()
        report = await harness.run()

        assert report.removed == ["corpus/doomed.txt"]
        assert "Doomed content" not in harness.indexed_texts()
        assert "Keep this content" in harness.indexed_texts()

        manifest = ManifestStore(harness.index_dir).load()
        assert "corpus/doomed.txt" not in manifest.entries
        # 元数据也要一并清掉，否则历史引用会指向一篇已不存在的文献
        from scitrace.service import SourceStore

        stored = SourceStore(harness.index_dir).load_sources()
        assert all(source.rel_path != "corpus/doomed.txt" for source in stored.values())
        assert keep.exists()

    async def test_renamed_file_is_delete_plus_add(self, harness: Harness) -> None:
        """改名 = 删除旧的 + 新增新的。这与用户直觉一致，

        而按内容哈希做键的实现会把改名误判为"未变化"而留下错误的路径信息。
        """
        path = harness.write("old-name.txt", "Same content, different filename.")
        await harness.run()
        path.rename(harness.corpus / "new-name.txt")

        report = await harness.run()
        assert report.removed == ["corpus/old-name.txt"]
        assert report.added == ["corpus/new-name.txt"]

    async def test_rebuild_ignores_manifest(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough for chunking.")
        await harness.run()

        report = await harness.run(rebuild=True)
        assert report.added == ["corpus/a.txt"]
        assert report.skipped == []
        assert len(harness.vector) == report.fragment_count


class TestFailureHandling:
    async def test_broken_file_does_not_block_others(self, harness: Harness) -> None:
        harness.write("good.txt", "Good content long enough for chunking.")
        (harness.corpus / "broken.pdf").write_bytes(b"%PDF-1.4\nnot actually a pdf")

        report = await harness.run()
        assert report.added == ["corpus/good.txt"]
        assert [name for name, _ in report.failed] == ["corpus/broken.pdf"]
        assert report.ok is False
        assert "无法打开" in report.failed[0][1]

    async def test_failed_file_is_retried_next_run(self, harness: Harness) -> None:
        """失败的文件即使哈希未变也要重试——用户可能刚补装了依赖或修好了文件。"""
        (harness.corpus / "broken.pdf").write_bytes(b"%PDF-1.4\nnope")
        first = await harness.run()
        assert first.failed

        second = await harness.run()
        assert [name for name, _ in second.failed] == ["corpus/broken.pdf"]
        assert second.skipped == []

    async def test_failed_entry_records_reason(self, harness: Harness) -> None:
        (harness.corpus / "broken.pdf").write_bytes(b"%PDF-1.4\nnope")
        await harness.run()
        entry = ManifestStore(harness.index_dir).load().entries["corpus/broken.pdf"]
        assert entry.status == "failed"
        assert entry.error

    async def test_scanned_pdf_reports_actionable_error(self, harness: Harness) -> None:
        from pdf_fixture import write_blank_pdf

        write_blank_pdf(harness.corpus / "scan.pdf")
        report = await harness.run()
        assert "扫描件" in report.failed[0][1]

    async def test_empty_corpus_is_fine(self, harness: Harness) -> None:
        report = await harness.run()
        assert report.added == [] and report.ok
        assert report.summary()


class TestIndexing:
    async def test_pdf_and_text_both_indexed(self, harness: Harness) -> None:
        write_text_pdf(harness.corpus / "paper.pdf", ["Abstract", "We study retrieval."])
        harness.write("notes.md", "# Notes\n\nSome notes about retrieval.")

        report = await harness.run()
        assert len(report.added) == 2
        assert len(harness.vector) == report.fragment_count
        assert len(harness.fulltext) == report.fragment_count

    async def test_embeddings_requested_per_fragment(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough for chunking.")
        report = await harness.run()
        embedded = sum(len(batch) for batch in harness.embedder.calls)
        assert embedded == report.fragment_count

    async def test_fulltext_is_committed(self, harness: Harness) -> None:
        """未 commit 的全文索写入对查询不可见——这是最容易漏掉的一步。"""
        harness.write("a.txt", "Alpha content long enough for chunking.")
        await harness.run()
        assert harness.fulltext.commit_calls >= 1
        assert harness.fulltext.pending_ids() == set()
        assert len(harness.fulltext) > 0

    async def test_vector_index_persisted(self, harness: Harness) -> None:
        harness.write("a.txt", "Alpha content long enough for chunking.")
        await harness.run()
        assert harness.vector.persist_calls >= 1

    async def test_metadata_resolver_is_consulted(self, harness: Harness) -> None:
        harness.resolver.patch = RESOLVED
        harness.write("a.txt", "Alpha content long enough for chunking.")

        report = await harness.run()
        source = next(iter(report.sources.values()))
        assert source.year == 2024
        assert source.venue == "Journal of Example Studies"
        assert harness.resolver.calls, "解析器必须被调用"

    async def test_source_metadata_persisted(self, harness: Harness) -> None:
        from scitrace.service import SourceStore

        harness.resolver.patch = RESOLVED
        harness.write("a.txt", "Alpha content long enough for chunking.")
        report = await harness.run()

        stored = SourceStore(harness.index_dir).load_sources()
        assert set(stored) == set(report.sources)
        assert next(iter(stored.values())).year == 2024

    async def test_doi_extracted_from_body(self, harness: Harness) -> None:
        """DOI 几乎只能从正文里找——有它才能走精确查询端点。"""
        harness.write("a.txt", "Title\n\nhttps://doi.org/10.5555/Example.2024.007\n\nBody text here that is long enough.")
        report = await harness.run()
        source = next(iter(report.sources.values()))
        assert source.doi == "10.5555/example.2024.007"

    async def test_arxiv_id_becomes_doi_when_no_doi_printed(self, harness: Harness) -> None:
        """预印本常只印 arXiv 编号而不印 DOI。

        arXiv 为每篇预印本分配固定格式的 DOI，据此构出 DOI 就能让元数据来源
        走**精确端点**。回归自实机验证：不加这一步时，一篇 arXiv 论文的标题
        补不上，文内引用退化成 ``(anonndpaperqa2 pages 2-3)``；
        加上之后变成 ``(skarlinski2024language pages 2-3)``。
        """
        harness.write(
            "preprint.txt",
            "Language agents achieve superhuman synthesis\n"
            "arXiv:2409.13740v1  [cs.AI]  10 Sep 2024\n\n"
            "Body content that is long enough to be chunked.",
        )
        report = await harness.run()
        source = next(iter(report.sources.values()))
        assert source.doi == "10.48550/arxiv.2409.13740"

    async def test_printed_doi_wins_over_arxiv_id(self, harness: Harness) -> None:
        """两者都出现时以印出来的 DOI 为准——它才是出版方分配的那个。"""
        harness.write(
            "both.txt",
            "Some paper\narXiv:2409.13740\nhttps://doi.org/10.5555/Real.2024.001\n\n"
            "Body content that is long enough to be chunked.",
        )
        report = await harness.run()
        assert next(iter(report.sources.values())).doi == "10.5555/real.2024.001"

    async def test_duplicate_doi_is_reported(self, harness: Harness) -> None:
        """同一 DOI 出现在两个文件里（预印本 + 正式版）会被正确归并，

        但必须让用户知道，否则"我索引了 2 篇，怎么只有 1 篇"无从解释。
        """
        body = "doi:10.5555/duplicate.2024.001\n\nSome body content that is long enough."
        harness.write("preprint.txt", body)
        harness.write("published.txt", body)

        report = await harness.run()
        assert len(report.sources) == 1
        assert report.duplicate_sources
        assert len(next(iter(report.duplicate_sources.values()))) == 2

    async def test_duplicate_doi_fragments_do_not_collide(self, harness: Harness) -> None:
        """两个文件共用一个 source_key 时，片段 id 必须仍然互不相同。

        这正是把文档内容哈希纳入 fragment_id 的原因：否则后摄入的一份会
        逐块覆盖前一份，索引里留下一个来源混杂的文档。
        """
        harness.write("preprint.txt", "doi:10.5555/dup.2024.001\n\nPreprint body content here.")
        harness.write("published.txt", "doi:10.5555/dup.2024.001\n\nPublished body content here.")
        await harness.run()

        texts = harness.indexed_texts()
        assert "Preprint body content" in texts
        assert "Published body content" in texts


class TestManifestStore:
    def test_missing_manifest_returns_empty(self, tmp_path: Path) -> None:
        assert ManifestStore(tmp_path / "nope").load().entries == {}

    def test_roundtrip(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        manifest = Manifest(entries={"corpus/a.txt": ManifestEntry(hash="abc", source_key="k")})
        store.save(manifest)
        assert store.load().entries["corpus/a.txt"].hash == "abc"

    def test_corrupt_manifest_degrades_to_rebuild(self, tmp_path: Path) -> None:
        """损坏的清单降级为全量重建，而不是让索引命令直接失败。"""
        store = ManifestStore(tmp_path)
        store.path.write_text("{not json", encoding="utf-8")
        assert store.load().entries == {}

    def test_schema_mismatch_degrades_to_rebuild(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        payload = {"schema_version": SCHEMA_VERSION + 5, "entries": {}}
        store.path.write_text(json.dumps(payload), encoding="utf-8")
        assert store.load().entries == {}

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        store = ManifestStore(tmp_path)
        store.save(Manifest(entries={}))
        assert store.path.is_file()
        assert not store.path.with_suffix(".json.tmp").exists()

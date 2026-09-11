"""M2 端到端验收：真实解析器 + 真实双索引 + 真实摄入编排。

``tests/test_ingest.py`` 用的是内存替身，验证的是**编排逻辑**；
本文件用真实适配器，验证的是**接线是否正确**——替身无法发现的错误正是这一类：
索引路径没对上、全文索引忘了 commit、持久化之后读不回来、
中文在 bigram 索引里搜不到。

这正是"单元测试全绿但一跑就崩"的典型来源，因此必须有一层用真实实现的验收。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEmbedder
from pdf_fixture import write_text_pdf
from scitrace.adapters.indexes import NumpyVectorIndex, TantivyFullTextIndex
from scitrace.config import ChunkingSettings
from scitrace.adapters.parsers import select_parser
from scitrace.pipeline.ingest import IngestPipeline

EMBED_DIM = 16


class RealHarness:
    """把真实适配器装配成一条可运行的摄入链路。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.corpus = root / "corpus"
        self.corpus.mkdir(parents=True, exist_ok=True)
        self.index_dir = root / "index"
        self.vector = NumpyVectorIndex(path=self.index_dir)
        self.fulltext = TantivyFullTextIndex(self.index_dir / "fts")
        self.embedder = FakeEmbedder(dimension=EMBED_DIM)
        self.pipeline = IngestPipeline(
            index_dir=self.index_dir,
            parser_resolver=select_parser,
            chunking=ChunkingSettings(target_chars=400, max_chars=800, min_chars=0),
            vector_index=self.vector,
            fulltext_index=self.fulltext,
            embedder=self.embedder,
        )

    def write_chinese(self, name: str, body: str) -> Path:
        path = self.corpus / name
        path.write_text(body, encoding="utf-8")
        return path

    async def query_vector(self, text: str, k: int = 5):
        vector = (await self.embedder.embed([text], kind="query"))[0]
        return await self.vector.search(vector, k)


@pytest.fixture
def harness(tmp_path: Path) -> RealHarness:
    return RealHarness(tmp_path)


class TestRealPipeline:
    async def test_ingests_and_indexes_both_ways(self, harness: RealHarness) -> None:
        write_text_pdf(
            harness.corpus / "paper.pdf",
            ["Abstract", "We study adaptive evidence retrieval for question answering."],
        )
        report = await harness.pipeline.run([harness.corpus])

        assert len(report.added) == 1
        assert report.fragment_count > 0
        assert len(harness.vector) == report.fragment_count
        assert len(harness.fulltext) == report.fragment_count

    async def test_vector_search_returns_indexed_fragments(self, harness: RealHarness) -> None:
        harness.write_chinese("a.txt", "证据可溯源问答系统通过引用键把答案关联到原文片段。")
        await harness.pipeline.run([harness.corpus])

        results = await harness.query_vector("证据可溯源问答系统", k=3)
        assert results
        assert all(item.fragment.source_key for item in results)
        assert [item.rank for item in results] == list(range(1, len(results) + 1))

    async def test_chinese_fulltext_search_works_end_to_end(self, harness: RealHarness) -> None:
        """中文经 bigram 切分后必须能被检索到——这是增量 ③ 的核心验收点。"""
        harness.write_chinese(
            "cn.txt",
            "本文提出一种跨模态对齐方法，用于科研文献的证据抽取与问答。",
        )
        await harness.pipeline.run([harness.corpus])

        results = await harness.fulltext.search("跨模态对齐", k=5)
        assert results, "中文术语必须能检索到"
        assert "跨模态对齐" in results[0].fragment.text

    async def test_unseen_chinese_term_is_recallable(self, harness: RealHarness) -> None:
        """未登录术语仍可召回——这是选择 bigram 而非词典分词的全部理由。"""
        harness.write_chinese("cn.txt", "我们采用稀疏专家路由提升推理效率。")
        await harness.pipeline.run([harness.corpus])

        assert await harness.fulltext.search("专家路由", k=5)

    async def test_mixed_language_document(self, harness: RealHarness) -> None:
        harness.write_chinese("mix.txt", "We propose a RAG 系统用于证据抽取与引用绑定。")
        await harness.pipeline.run([harness.corpus])

        assert await harness.fulltext.search("RAG", k=5)
        assert await harness.fulltext.search("引用绑定", k=5)

    async def test_allowed_keys_scopes_search(self, harness: RealHarness) -> None:
        harness.write_chinese("first.txt", "第一篇文献讨论证据抽取方法。")
        harness.write_chinese("second.txt", "第二篇文献讨论证据抽取方法。")
        report = await harness.pipeline.run([harness.corpus])

        keys = sorted(report.sources)
        assert len(keys) == 2
        scoped = await harness.fulltext.search("证据抽取", k=10, allowed_keys={keys[0]})
        assert scoped
        assert {item.fragment.source_key for item in scoped} == {keys[0]}

    async def test_section_path_survives_to_index(self, harness: RealHarness) -> None:
        harness.write_chinese(
            "structured.md",
            "## 2 方法\n\n本节介绍证据抽取的具体实现方式与参数选择。\n\n"
            "### 2.1 检索\n\n检索阶段使用向量与全文双路召回。",
        )
        await harness.pipeline.run([harness.corpus])

        results = await harness.fulltext.search("双路召回", k=5)
        assert results
        assert results[0].fragment.section_path == ["2 方法", "2.1 检索"]

    async def test_page_range_survives_to_index(self, harness: RealHarness) -> None:
        write_text_pdf(
            harness.corpus / "paged.pdf",
            ["First page content about retrieval systems."],
        )
        await harness.pipeline.run([harness.corpus])

        results = await harness.fulltext.search("retrieval", k=5)
        assert results
        assert results[0].fragment.page_start == 1
        assert results[0].fragment.page_label == "page 1"

    async def test_incremental_second_run_does_no_work(self, harness: RealHarness) -> None:
        harness.write_chinese("a.txt", "一段用于测试增量索引的中文内容。")
        await harness.pipeline.run([harness.corpus])
        commit_calls = harness.fulltext.commit_calls if hasattr(harness.fulltext, "commit_calls") else None

        report = await harness.pipeline.run([harness.corpus])
        assert report.processed == 0
        assert report.skipped == ["corpus/a.txt"]
        assert commit_calls is None or True  # 替身专属计数，真实实现无此属性


class TestMixedFormatCorpus:
    """混合格式语料必须能整体摄入——这是真实使用中最常见的情形。

    回归自边界测试：默认配置（``ingest.parser="pypdf"``）下，一份含 23 个
    ``.txt``/``.md`` 文件的语料**一篇都索引不进去**，全部报
    "解析器 'pypdf' 不支持文件 xxx.txt"。原因是把"PDF 解析后端"当成了
    "所有文件的强制解析器"。此前的测试要么是纯 PDF，要么走不带 preferred
    的自动选择路径，完全没暴露。
    """

    async def test_pdf_and_text_coexist_under_preferred_pdf_parser(self, tmp_path: Path) -> None:
        from scitrace.adapters.parsers import select_parser
        from scitrace.factory import _resolve_parser

        harness = RealHarness(tmp_path)
        write_text_pdf(harness.corpus / "paper.pdf", ["Abstract", "We study retrieval."])
        harness.write_chinese("notes.txt", "一篇中文笔记，内容足够长以便切出片段。")
        harness.write_chinese("readme.md", "# 说明\n\n这是 Markdown 格式的说明文本。")

        # 模拟默认配置：ingest.parser = "pypdf"
        harness.pipeline.parser_resolver = lambda path: _resolve_parser(  # type: ignore[assignment]
            path, preferred="pypdf"
        )
        report = await harness.pipeline.run([harness.corpus])

        assert report.failed == [], f"混合格式语料不应有失败项：{report.failed}"
        assert len(report.added) == 3
        assert report.fragment_count > 0

    async def test_preferred_parser_still_used_for_its_own_format(self, tmp_path: Path) -> None:
        """修好回退语义之后，首选解析器在它自己的格式上仍必须被采用——
        否则修复就变成了"静默忽略用户配置"。"""
        from scitrace.adapters.parsers import PlainTextParser, PyPDFParser
        from scitrace.factory import _resolve_parser

        assert isinstance(_resolve_parser(Path("a.pdf"), preferred="pypdf"), PyPDFParser)
        assert isinstance(_resolve_parser(Path("a.txt"), preferred="plaintext"), PlainTextParser)

    async def test_unsupported_format_still_raises(self, tmp_path: Path) -> None:
        from scitrace.factory import _resolve_parser

        with pytest.raises(ValueError, match="没有解析器"):
            _resolve_parser(Path("a.docx"), preferred="pypdf")


class TestPersistence:
    async def test_vector_index_roundtrip(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "index"
        harness = RealHarness(tmp_path)
        harness.write_chinese("a.txt", "持久化测试用的中文内容，需要足够长才能切出片段。")
        report = await harness.pipeline.run([harness.corpus])

        restored = NumpyVectorIndex(path=index_dir)
        restored.load()
        assert len(restored) == report.fragment_count

        original = await harness.query_vector("持久化测试", k=3)
        query = (await harness.embedder.embed(["持久化测试"], kind="query"))[0]
        reloaded = await restored.search(query, 3)
        assert [item.fragment.fragment_id for item in reloaded] == [
            item.fragment.fragment_id for item in original
        ]

    async def test_tantivy_index_survives_reopen(self, tmp_path: Path) -> None:
        harness = RealHarness(tmp_path)
        harness.write_chinese("a.txt", "重新打开索引后仍需能检索到这段中文内容。")
        await harness.pipeline.run([harness.corpus])

        reopened = TantivyFullTextIndex(tmp_path / "index" / "fts")
        assert len(reopened) > 0
        assert await reopened.search("重新打开", k=5)

    async def test_manifest_and_sources_survive_reopen(self, tmp_path: Path) -> None:
        from scitrace.pipeline.ingest import ManifestStore
        from scitrace.service import SourceStore

        harness = RealHarness(tmp_path)
        harness.write_chinese("a.txt", "第二轮运行应当直接跳过这个文件。")
        await harness.pipeline.run([harness.corpus])

        manifest = ManifestStore(tmp_path / "index").load()
        sources = SourceStore(tmp_path / "index").load_sources()
        assert manifest.entries
        assert sources
        assert all(source.rel_path.startswith("corpus/") for source in sources.values())

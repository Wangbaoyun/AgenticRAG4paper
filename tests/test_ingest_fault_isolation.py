"""故障注入：验证任何一个阶段的失败都只影响那一个文件。

## 为什么必须有这个文件

真实语料曾经让 36 篇论文的索引**整批作废**：PDF 解析出的孤立代理项让 tantivy
编码失败，异常冲出 worker、掀翻整个 anyio task group。

当时的修法是给 `_index_fragments` 单独加 try——但那只把下一次事故推迟到
下一个未加保护的语句（事后核查发现 `_unindex_source` 就仍在保护之外）。
**"补了一个调用点"如果没有覆盖全部阶段的测试，就没有任何证据支撑。**

本文件对摄入的**每一个阶段**各注入一次异常，断言其余文件仍然全部完成。
它防的不是某一个已知 bug，而是"task group 的任何一个裸 await 都是整批作废的机会"
这一结构性风险。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_ingest import Harness

GOOD_FILES = 4
BAD_FILE = "bad.txt"


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    """每个测试一份独立的语料、索引与替身。

    刻意在本文件内重新定义而非从 ``test_ingest`` 导入 fixture：
    fixture 不随类的导入而传递，跨模块隐式依赖会让本文件的独立性变差。
    """
    return Harness(tmp_path)


def seed(harness: Harness) -> Path:
    """写入 4 个正常文件与 1 个将被注入故障的文件。"""
    for index in range(GOOD_FILES):
        harness.write(f"good{index}.txt", f"Content of good file {index}, long enough to chunk.")
    return harness.write(BAD_FILE, "Content of the target file, long enough to be chunked.")


#: 被注入故障的那个文件的内容标记。
#:
#: 索引阶段拿到的是 ``Fragment`` 与 ``source_key``（哈希），**拿不到文件名**——
#: 所以这里按文本内容定位，而不是按文件名。
TARGET_MARKER = "target file"


class Saboteur:
    """让指定对象在指定阶段抛出异常的包装器。"""

    def __init__(self, marker: str = TARGET_MARKER, message: str = "injected failure") -> None:
        self.marker = marker
        self.message = message
        self.triggered = False

    def is_target(self, *candidates: object) -> bool:
        return any(self.marker in str(item) for item in candidates)

    def blow_up(self) -> None:
        self.triggered = True
        raise RuntimeError(self.message)


class TestParseStage:
    async def test_parse_failure_isolated(self, harness: Harness) -> None:
        seed(harness)
        original = harness.pipeline.ingest_file
        saboteur = Saboteur(BAD_FILE)

        async def patched(path: Path, *, identifier: str, digest: str):
            if saboteur.is_target(path.name):
                saboteur.blow_up()
            return await original(path, identifier=identifier, digest=digest)

        harness.pipeline.ingest_file = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert saboteur.triggered
        assert len(report.added) == GOOD_FILES, "其余文件必须全部完成"
        assert [name for name, _ in report.failed] == [f"corpus/{BAD_FILE}"]


class TestIndexStage:
    async def test_vector_index_failure_isolated(self, harness: Harness) -> None:
        seed(harness)
        original = harness.vector.add
        saboteur = Saboteur()   # 按内容标记定位，索引阶段拿不到文件名

        async def patched(items):
            if saboteur.is_target(*(fragment.text for fragment, _ in items)):
                saboteur.blow_up()
            return await original(items)

        harness.vector.add = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert saboteur.triggered
        assert len(report.added) == GOOD_FILES
        assert len(report.failed) == 1

    async def test_fulltext_index_failure_isolated(self, harness: Harness) -> None:
        """这条正是真实语料崩掉的那条路径（tantivy 编码失败）。"""
        seed(harness)
        original = harness.fulltext.add
        saboteur = Saboteur()   # 同上：索引阶段拿不到文件名

        async def patched(fragments, *, titles=None):
            if saboteur.is_target(*(item.text for item in fragments)):
                saboteur.blow_up()
            return await original(fragments, titles=titles)

        harness.fulltext.add = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert saboteur.triggered
        assert len(report.added) == GOOD_FILES
        assert len(report.failed) == 1

    async def test_embedder_failure_isolated(self, harness: Harness) -> None:
        seed(harness)
        original = harness.embedder.embed
        saboteur = Saboteur()

        async def patched(texts, *, kind="document"):
            if saboteur.is_target(*texts):
                saboteur.blow_up()
            return await original(texts, kind=kind)

        harness.embedder.embed = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert saboteur.triggered
        assert len(report.added) == GOOD_FILES


class TestUnindexStage:
    """更新路径：`_unindex_source` 此前就裸露在 try 之外。"""

    async def test_unindex_failure_isolated(self, harness: Harness) -> None:
        target = seed(harness)
        first = await harness.run()
        assert len(first.added) == GOOD_FILES + 1

        # 改内容触发更新，从而走到 _unindex_source
        target.write_text("Brand new content for the target file, long enough.", encoding="utf-8")

        original = harness.pipeline._unindex_source
        saboteur = Saboteur()

        async def patched(source_key: str) -> None:
            # 只对"bad.txt"对应的 source_key 注入——它是本轮唯一需要更新清理的文件
            if source_key == next(
                key
                for key, value in harness.pipeline.source_store.load_sources().items()
                if BAD_FILE in value.rel_path
            ):
                saboteur.blow_up()
            return await original(source_key)

        harness.pipeline._unindex_source = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert saboteur.triggered, "本测试必须真的走到 _unindex_source"
        assert len(report.failed) == 1
        assert f"corpus/{BAD_FILE}" in report.failed[0][0]


class TestStoreStage:
    """清单与元数据落盘失败**应当**让整次摄入失败——这条路不该被隔离。

    理由：清单与索引不一致比整次失败更危险。清单说"这些文件已索引"而索引里没有，
    下次运行会跳过它们，于是索引永久性地缺内容，且没有任何迹象。
    """

    async def test_manifest_save_failure_fails_the_run(self, harness: Harness) -> None:
        seed(harness)
        original = harness.pipeline.manifest_store.save

        def patched(manifest):
            raise OSError("disk full")

        harness.pipeline.manifest_store.save = patched  # type: ignore[method-assign]
        with pytest.raises(OSError, match="disk full"):
            await harness.run()
        harness.pipeline.manifest_store.save = original  # type: ignore[method-assign]


class TestNoStageIsLeftUnguarded:
    async def test_all_good_files_survive_any_single_failure(self, harness: Harness) -> None:
        """把 5 个文件里的 1 个在所有阶段都设为失败，其余 4 个必须全部成功。"""
        seed(harness)
        original_parse = harness.pipeline.ingest_file
        saboteur = Saboteur(BAD_FILE)

        async def patched(path: Path, *, identifier: str, digest: str):
            if saboteur.is_target(path.name):
                saboteur.blow_up()
            return await original_parse(path, identifier=identifier, digest=digest)

        harness.pipeline.ingest_file = patched  # type: ignore[method-assign]
        report = await harness.run()

        assert report.summary()
        assert len(report.added) == GOOD_FILES
        assert harness.vector.__len__() > 0, "正常文件的片段必须真的进了索引"
        assert len(harness.fulltext) > 0
        assert not any(BAD_FILE in name for name in report.added)

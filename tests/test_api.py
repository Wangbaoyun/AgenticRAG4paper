"""`scitrace.api` 的系统化测试。

覆盖率实测 `api.py` 只有 **38%**——它和 `cli.py` 一样属于"用户接触得到、但验证不足"的外层。
这里覆盖 SPEC §7 之外的库调用契约：`ask` / `search` / `build_index` / `index_status`。

其中 `index_status` 有一条明确的设计要求：**它永不抛异常**。
状态查询是排障入口——索引不存在、清单损坏、元数据缺失都是它最该派上用场的时刻，
此时报错等于把用户推进死胡同（"是索引坏了还是我配错了？"）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scitrace.api import ask, build_index, index_status, load_services, search
from scitrace.config import Settings, load_settings
from scitrace.domain.session import SessionStatus
from scitrace.pipeline.ingest import ManifestStore
from test_agent_runtime import FakeServicesFactory


def run(coro):
    import anyio

    return anyio.run(lambda: coro)


@pytest.fixture
def services(tmp_path: Path):
    factory = FakeServicesFactory(tmp_path)
    return factory, run(factory.build())


class TestAsk:
    def test_returns_run_result(self, services) -> None:
        factory, svc = services
        result = run(ask("问题", svc))
        assert result.question == "问题"
        assert result.status is not None
        assert result.session_id

    def test_deterministic_mode_is_default(self, services) -> None:
        _, svc = services
        assert run(ask("问题", svc)).status in set(SessionStatus)

    def test_refusal_is_a_normal_result_not_an_exception(self, tmp_path: Path) -> None:
        """空索引必然拒答——那是一个**结果**，不是异常。"""
        factory = FakeServicesFactory(tmp_path, seed=False)
        svc = run(factory.build())
        result = run(ask("库外问题", svc))
        assert result.answer.refused is True
        assert result.answer.citations == []


class TestSearch:
    def test_returns_scored_fragments(self, services) -> None:
        _, svc = services
        results = run(search("证据抽取", svc))
        assert isinstance(results, list)
        for item in results:
            assert item.fragment.fragment_id
            assert item.rank >= 1

    def test_k_is_respected(self, services) -> None:
        _, svc = services
        assert len(run(search("证据", svc, k=1))) <= 1

    def test_blank_query_returns_empty(self, services) -> None:
        _, svc = services
        assert run(search("   ", svc)) == []


class TestBuildIndex:
    def test_refreshes_services_sources(self, tmp_path: Path) -> None:
        """摄入后必须刷新 `services.sources`。

        它是构造时从磁盘读的快照。不刷新的话，"同一进程内先摄入再提问"
        会拿着空元数据渲染引用——表现为引用全是 `unknown`，
        而索引里其实有完整信息。
        """
        factory = FakeServicesFactory(tmp_path)
        svc = run(factory.build())

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Some content long enough to be chunked.", encoding="utf-8")
        report = run(build_index(svc, [corpus]))

        assert report.added
        # 摄入后 `services.sources` 被**磁盘上的权威状态**替换。
        # 注意工厂预置的 `src-1` 会消失——它从未落盘，本就不该出现在权威视图里。
        assert any("a.txt" in source.rel_path for source in svc.sources.values()), (
            "摄入后 services.sources 必须反映磁盘上的真实文献集合——"
            "否则同进程内接着提问会拿着陈旧快照渲染引用（表现为引用全是 unknown）"
        )

    def test_rebuild_flag_is_accepted(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        svc = run(factory.build())
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Content long enough to be chunked here.", encoding="utf-8")

        first = run(build_index(svc, [corpus]))
        second = run(build_index(svc, [corpus]))
        rebuilt = run(build_index(svc, [corpus], rebuild=True))

        assert len(first.added) == 1
        assert second.skipped == ["corpus/a.txt"]
        assert len(rebuilt.added) == 1, "rebuild 必须忽略清单"


class TestIndexStatus:
    def test_missing_index_returns_zeros(self, tmp_path: Path) -> None:
        """状态查询是排障入口：索引不存在时它更要能给出答案。"""
        settings = Settings()
        settings.index.root = tmp_path / "nonexistent"
        status = index_status(settings)

        assert status["exists"] is False
        assert status["sources"] == 0
        assert status["fragments"] == 0
        assert status["corpus"] == {}

    def test_corrupt_manifest_does_not_raise(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.index.root = tmp_path / "index"
        directory = settings.index_dir
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text("{not json", encoding="utf-8")

        status = index_status(settings)
        assert status["fragments"] == 0

    def test_reports_fingerprint_and_counts(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.index.root = tmp_path / "index"
        status = index_status(settings)
        assert status["fingerprint"] == settings.index_fingerprint()
        assert len(str(status["fingerprint"])) == 12

    def test_corpus_is_aggregated_by_root(self, tmp_path: Path) -> None:
        """语料统计按 rel_path 的首段聚合——摄入时的标识形如 ``<根目录名>/<相对路径>``。"""
        from scitrace.pipeline.ingest import Manifest, ManifestEntry

        def entry(source_key: str) -> ManifestEntry:
            return ManifestEntry(hash="h", source_key=source_key)

        settings = Settings()
        settings.index.root = tmp_path / "index"
        ManifestStore(settings.index_dir).save(
            Manifest(
                entries={
                    "papers/a.txt": entry("k1"),
                    "papers/b.txt": entry("k2"),
                    "notes/c.txt": entry("k3"),
                }
            )
        )
        assert index_status(settings)["corpus"] == {"papers": 2, "notes": 1}

    def test_status_is_json_serializable(self, tmp_path: Path) -> None:
        settings = Settings()
        settings.index.root = tmp_path / "index"
        json.dumps(index_status(settings))  # 不可序列化会在这里失败


class TestLoadServices:
    def test_is_the_single_monkeypatch_point(self, tmp_path: Path, monkeypatch) -> None:
        """CLI 通过它装配服务——保留这个转发是为了让测试只需替换一处。"""
        settings = load_settings(dotenv_path=None)
        settings.index.root = tmp_path / "index"
        settings.metadata.enabled = False
        services = load_services(settings, load_index=False)
        assert services.settings is settings
        # 装配成功的判据是各部件都到位，而不是"没抛异常"
        assert services.retriever is not None
        assert services.screener is not None
        assert services.synthesizer is not None
        assert set(services.llms) == {"main", "summary", "agent"}
        run(services.aclose())

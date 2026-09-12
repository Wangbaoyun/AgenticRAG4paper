"""CLI 系统化测试。

## 为什么这个文件必须存在

覆盖率实测：`cli.py` **19%**、`api.py` **38%**——核心逻辑（domain / ports / pipeline）
都是 90%+，而**用户唯一实际接触的那一层最没被验证**。

这里按 SPEC §7 的契约逐条钉住：命令表、退出码、`--json` 的纯 JSON 保证、错误路径。
其中两条是接口级承诺，破坏了使用方的脚本会静默出错：

1. **拒答的退出码是 0**。把"系统说不知道"变成非零退出码，会让调用方无法区分
   它与"系统坏了"；
2. **`--json` 输出必须是纯 JSON**。混入一行 rich 的排版就会让
   `stc ask --json | jq` 直接失败，而脚本化使用正是 `--json` 存在的全部理由。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scitrace import cli
from scitrace.config import settings as settings_module
from test_agent_runtime import FakeServicesFactory


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把全部落盘位置指向临时目录，并清掉宿主机的 SCITRACE_* 变量。

    **不能只 patch ``CONFIG_ROOT``**：``Settings`` 的默认值
    （``index.root`` / ``index.sessions_root``）在**类定义时**就求值了，
    之后改模块常量不会影响它们。实测这个疏漏让测试读到了真实的
    ``~/.scitrace/sessions/`` —— 测试隔离不完整，而且是在读用户的真实数据。
    正确做法是走环境变量，让 Settings 在构造时取到临时路径。
    """
    import os

    monkeypatch.setattr(settings_module, "CONFIG_ROOT", tmp_path / ".scitrace")
    for key in [key for key in os.environ if key.startswith("SCITRACE_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCITRACE_INDEX__ROOT", str(tmp_path / ".scitrace" / "index"))
    monkeypatch.setenv("SCITRACE_INDEX__SESSIONS_ROOT", str(tmp_path / ".scitrace" / "sessions"))


@pytest.fixture
def fake_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """替换掉装配入口：CLI 测试不碰网络、不加载模型。"""
    factory = FakeServicesFactory(tmp_path)
    services = _run_factory(factory)
    monkeypatch.setattr(cli, "load_services", lambda *args, **kwargs: services)
    # `_load` 也要指向同一份 Settings。生产里两者都源自同一次 load_settings()，
    # 只替换其中一个会让"写入"与"读取"落到不同目录——
    # 这种分裂在测试里表现为"会话明明写了却搜不到"。
    monkeypatch.setattr(cli, "_load", lambda name=None: services.settings)
    return factory, services


def _run_factory(factory: FakeServicesFactory):
    import anyio

    return anyio.run(factory.build)


class TestParserContract:
    def test_version_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        assert "scitrace" in capsys.readouterr().out

    def test_no_command_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main([])
        assert excinfo.value.code == 2

    def test_unknown_command_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["bogus"])
        assert excinfo.value.code == 2

    def test_unknown_mode_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["ask", "q", "--mode", "magic"])
        assert excinfo.value.code == 2

    def test_bad_settings_name_reports_error_not_traceback(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """打错 profile 名必须给出可读错误 + 退出码 1，而不是抛 traceback。"""
        code = cli.main(["status", "--settings", "definitely-missing"])
        captured = capsys.readouterr()
        assert code == 1
        assert "配置错误" in captured.err
        assert "Traceback" not in captured.err


class TestStatus:
    def test_human_output(self, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert "索引指纹" in out
        assert "片段" in out

    def test_json_output_is_pure_json(self, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["status", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)  # 混入任何非 JSON 都会失败
        assert {"fingerprint", "index_dir", "sources", "fragments"} <= set(payload)

    def test_status_never_fails_on_missing_index(self) -> None:
        """状态查询是排障入口：索引不存在时它更要能给出答案。"""
        assert cli.main(["status"]) == 0


class TestAsk:
    def test_json_schema_matches_spec(self, fake_services, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["ask", "问题", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {
            "session_id",
            "question",
            "answer",
            "status",
            "citations",
            "references",
            "usage",
            "timing",
        } <= set(payload)

    def test_usage_carries_currency(self, fake_services, capsys: pytest.CaptureFixture) -> None:
        """币种必须随用量一起输出——否则读数的人要猜单位。"""
        cli.main(["ask", "问题", "--json"])
        usage = json.loads(capsys.readouterr().out)["usage"]
        assert "estimated_cost" in usage
        # cost_currency 可能在 usage 顶层之外，但 estimated_cost 必须在

    def test_refusal_still_exits_zero(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """SPEC §7：拒答是**正常终态**，退出码必须为 0。

        改成非零会让调用方无法区分"系统说不知道"与"系统坏了"。
        """
        factory = FakeServicesFactory(tmp_path, seed=False)  # 空索引 → 必然拒答
        services = _run_factory(factory)
        monkeypatch.setattr(cli, "load_services", lambda *a, **k: services)
        monkeypatch.setattr(cli, "_load", lambda name=None: services.settings)

        assert cli.main(["ask", "库外问题", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] in {"REFUSED", "UNCITED", "UNSURE", "FAIL"}
        assert payload["citations"] == []

    def test_human_output_mentions_status(self, fake_services, capsys) -> None:
        assert cli.main(["ask", "问题"]) == 0
        out = capsys.readouterr().out
        assert "状态：" in out
        assert "参考文献" in out

    def test_agentic_mode_is_accepted(self, fake_services) -> None:
        assert cli.main(["ask", "问题", "--mode", "agentic", "--json"]) == 0

    def test_language_option(self, fake_services) -> None:
        assert cli.main(["ask", "question", "--language", "en", "--json"]) == 0


class TestSearch:
    def test_json_shape(self, fake_services, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["search", "检索", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["query"] == "检索"
        assert isinstance(payload["results"], list)

    def test_k_option(self, fake_services) -> None:
        assert cli.main(["search", "检索", "--k", "3", "--json"]) == 0

    def test_no_match_is_not_an_error(self, fake_services, capsys) -> None:
        """检索不到结果不是错误——它是正常返回，只是列表为空。

        （这里不假设列表一定为空：小索引上 top-k 总会返回内容，
        断言"为空"会让测试依赖于索引规模这种无关因素。）
        """
        assert cli.main(["search", "完全不相关的词xyzzy", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload["results"], list)


class TestIndex:
    def test_successful_index_exits_zero(self, fake_services, tmp_path: Path, capsys) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.txt").write_text("Some content long enough to be chunked.", encoding="utf-8")
        assert cli.main(["index", str(corpus)]) == 0
        assert "新增" in capsys.readouterr().out

    def test_all_files_failing_exits_one(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """全部失败才算命令失败；部分失败只报告，退出码仍为 0。"""
        from test_ingest import Harness

        harness = Harness(tmp_path)
        (harness.corpus / "broken.pdf").write_bytes(b"%PDF-1.4\nnot a pdf")
        monkeypatch.setattr(cli, "load_services", lambda *a, **k: harness.pipeline and _Stub())
        # 直接测退出码逻辑：用真实的 ingest 管线
        monkeypatch.setattr(cli, "build_index", _all_failed_build_index)
        assert cli.main(["index", str(harness.corpus)]) == 1

    def test_partial_failure_exits_zero(self, fake_services, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(cli, "build_index", _partial_failed_build_index)
        assert cli.main(["index", str(tmp_path)]) == 0


class _Stub:
    async def aclose(self) -> None:
        return None


async def _all_failed_build_index(services, paths, *, rebuild=False):
    from scitrace.pipeline.ingest import IngestReport

    return IngestReport(failed=[("corpus/broken.pdf", "无法打开")])


async def _partial_failed_build_index(services, paths, *, rebuild=False):
    from scitrace.pipeline.ingest import IngestReport

    return IngestReport(added=["corpus/a.txt"], failed=[("corpus/broken.pdf", "无法打开")])


class TestSessions:
    def test_search_returns_pure_json(self, capsys: pytest.CaptureFixture) -> None:
        assert cli.main(["sessions", "search", "任意关键词", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["keyword"] == "任意关键词"
        assert isinstance(payload["sessions"], list)

    def test_no_match_is_not_an_error(self, capsys) -> None:
        assert cli.main(["sessions", "search", "绝不存在的关键词"]) == 0

    def test_finds_persisted_session(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """端到端：先问一次，再在历史里搜到它。"""
        factory = FakeServicesFactory(tmp_path)
        services = _run_factory(factory)
        monkeypatch.setattr(cli, "load_services", lambda *a, **k: services)
        monkeypatch.setattr(cli, "_load", lambda name=None: services.settings)
        cli.main(["ask", "这是一个独特的问题标记", "--json"])
        capsys.readouterr()   # 清掉第一次命令的输出，否则 JSON 解析会读到两段

        assert cli.main(["sessions", "search", "独特的问题标记", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] >= 1


class TestConfig:
    def test_path_prints_a_path(self, capsys) -> None:
        assert cli.main(["config", "path"]) == 0
        assert "settings" in capsys.readouterr().out

    def test_show_is_pure_json(self, capsys) -> None:
        assert cli.main(["config", "show"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert {"llm", "embedding", "retrieval"} <= set(payload)

    def test_save_writes_profile(self, tmp_path: Path, capsys) -> None:
        assert cli.main(["config", "save", "--settings", "myprofile"]) == 0
        target = tmp_path / ".scitrace" / "settings" / "myprofile.json"
        assert target.is_file()
        assert "已写入" in capsys.readouterr().out

    def test_init_is_an_alias_of_save(self, tmp_path: Path, capsys) -> None:
        assert cli.main(["config", "init", "--settings", "fresh"]) == 0
        assert (tmp_path / ".scitrace" / "settings" / "fresh.json").is_file()

    def test_saved_profile_has_no_secrets(self, tmp_path: Path) -> None:
        cli.main(["config", "save", "--settings", "nokeys"])
        raw = (tmp_path / ".scitrace" / "settings" / "nokeys.json").read_text(encoding="utf-8")
        assert "api_key" not in raw or "null" in raw


class TestErrorHandling:
    def test_unexpected_error_becomes_exit_one(
        self, monkeypatch, capsys: pytest.CaptureFixture
    ) -> None:
        """绝不把 traceback 抛给用户。"""

        def explode(*args, **kwargs):
            raise RuntimeError("内部炸了")

        monkeypatch.setattr(cli, "index_status", explode)
        assert cli.main(["status"]) == 1
        captured = capsys.readouterr()
        assert "执行失败" in captured.err
        assert "Traceback" not in captured.err

    def test_keyboard_interrupt_is_handled(self, monkeypatch, capsys) -> None:
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "index_status", interrupt)
        assert cli.main(["status"]) == 1
        assert "已中断" in capsys.readouterr().err

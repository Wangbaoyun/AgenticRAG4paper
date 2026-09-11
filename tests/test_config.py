"""测试配置系统：五层优先级、类型校验与索引指纹。

配置错误的表现形式往往是"实验跑完了但结果是错的"（用了默认模型、索引名对不上），
而不是抛异常。因此这里把优先级链的**每一层**都单独钉一遍。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from scitrace import SCHEMA_VERSION
from scitrace.config import (
    ChunkingSettings,
    IngestSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    SettingsError,
    deep_merge,
    env_to_nested,
    load_settings,
    read_dotenv,
    save_named,
    settings_path,
    sorted_available_profiles,
)
from scitrace.config import settings as settings_module


@pytest.fixture(autouse=True)
def _isolate_scitrace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清空进程中的 SCITRACE_* 变量，避免宿主环境影响测试结果。"""
    for key in [key for key in os.environ if key.startswith("SCITRACE_")]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把配置根目录重定向到临时目录，避免污染真实的 ~/.scitrace。"""
    root = tmp_path / ".scitrace"
    monkeypatch.setattr(settings_module, "CONFIG_ROOT", root)
    return root


def write_profile(root: Path, name: str, payload: dict) -> Path:
    path = root / "settings" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestDefaults:
    def test_settings_constructible_without_config(self) -> None:
        settings = Settings()
        assert settings.llm.model
        assert settings.embedding.model
        assert settings.retrieval.strategy == "dense_mmr"
        assert settings.screening.min_relevance == 5
        assert settings.agent.max_steps == 12

    def test_unknown_key_is_rejected(self, config_root: Path) -> None:
        """extra='forbid'：拼错的字段必须报错，不能静默生效为默认值。"""
        write_profile(config_root, "bad", {"llm": {"modle": "typo"}})
        with pytest.raises(SettingsError, match="配置校验失败"):
            load_settings(name="bad", dotenv_path=None)


class TestFiveTierPrecedence:
    """SPEC §5.1：显式覆盖 > 进程环境变量 > 命名 profile > .env > 默认。"""

    def test_tier1_defaults(self, config_root: Path) -> None:
        assert load_settings(dotenv_path=None).llm.model == "deepseek/deepseek-chat"

    def test_tier2_dotenv_overrides_default(self, config_root: Path, tmp_path: Path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("SCITRACE_LLM__MODEL=from-dotenv\n", encoding="utf-8")
        assert load_settings(dotenv_path=dotenv).llm.model == "from-dotenv"

    def test_tier3_profile_overrides_dotenv(self, config_root: Path, tmp_path: Path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("SCITRACE_LLM__MODEL=from-dotenv\n", encoding="utf-8")
        write_profile(config_root, "p", {"llm": {"model": "from-profile"}})
        assert load_settings(name="p", dotenv_path=dotenv).llm.model == "from-profile"

    def test_tier4_process_env_overrides_profile(
        self, config_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CI 里临时覆盖一次实验，不应被仓库中的 profile 压过去。"""
        write_profile(config_root, "p", {"llm": {"model": "from-profile"}})
        monkeypatch.setenv("SCITRACE_LLM__MODEL", "from-process-env")
        assert load_settings(name="p", dotenv_path=None).llm.model == "from-process-env"

    def test_tier5_overrides_beat_everything(
        self, config_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("SCITRACE_LLM__MODEL=from-dotenv\n", encoding="utf-8")
        write_profile(config_root, "p", {"llm": {"model": "from-profile"}})
        monkeypatch.setenv("SCITRACE_LLM__MODEL", "from-process-env")

        settings = load_settings(
            name="p", dotenv_path=dotenv, overrides={"llm": {"model": "from-cli"}}
        )
        assert settings.llm.model == "from-cli"

    def test_partial_profile_does_not_reset_other_fields(self, config_root: Path) -> None:
        """深层合并：profile 只覆盖它提到的字段，其余保持默认。"""
        write_profile(config_root, "p", {"llm": {"model": "x"}})
        settings = load_settings(name="p", dotenv_path=None)
        assert settings.llm.model == "x"
        assert settings.llm.max_tokens == 4096  # 未被 profile 触及


class TestProfileHandling:
    def test_missing_explicit_profile_raises(self, config_root: Path) -> None:
        """打错一个字母就用默认配置跑完实验，是最难发现的一类错误。"""
        with pytest.raises(SettingsError, match="不存在"):
            load_settings(name="typo", dotenv_path=None)

    def test_missing_default_profile_is_fine(self, config_root: Path) -> None:
        assert load_settings(name=None, dotenv_path=None).llm.model

    def test_invalid_json_raises(self, config_root: Path) -> None:
        path = config_root / "settings" / "broken.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SettingsError, match="合法 JSON"):
            load_settings(name="broken", dotenv_path=None)

    def test_non_object_top_level_raises(self, config_root: Path) -> None:
        write_profile(config_root, "list", [1, 2])  # type: ignore[arg-type]
        with pytest.raises(SettingsError, match="必须是对象"):
            load_settings(name="list", dotenv_path=None)

    def test_settings_path_and_listing(self, config_root: Path) -> None:
        write_profile(config_root, "alpha", {})
        write_profile(config_root, "beta", {})
        assert settings_path("alpha").name == "alpha.json"
        assert sorted_available_profiles() == ["alpha", "beta"]

    def test_save_then_load_roundtrip(self, config_root: Path) -> None:
        settings = Settings()
        settings.retrieval.strategy = "hybrid_rrf"
        settings.retrieval.k = 25
        save_named(settings, "saved")

        reloaded = load_settings(name="saved", dotenv_path=None)
        assert reloaded.retrieval.strategy == "hybrid_rrf"
        assert reloaded.retrieval.k == 25

    def test_saved_profile_never_contains_secrets(self, config_root: Path) -> None:
        """profile 会被提交进仓库用于复现，密钥必须留在 .env 里。

        因此刻意**不提供**"写入密钥"的开关——一个会被误用的开关不如没有。
        """
        settings = Settings()
        settings.llm.api_key = "sk-super-secret"  # type: ignore[assignment]
        settings.metadata.semantic_scholar_api_key = "s2-secret"  # type: ignore[assignment]
        save_named(settings, "withkey")

        raw = settings_path("withkey").read_text(encoding="utf-8")
        assert "sk-super-secret" not in raw
        assert "s2-secret" not in raw
        assert "SecretStr" not in raw

    def test_masking_is_type_based_not_name_based(self, config_root: Path) -> None:
        """回归：按字段名含 'key'/'token' 遮罩会误伤 max_tokens。

        早先的实现把 ``llm.max_tokens`` 与 ``agent.max_tokens`` 置为 None，
        导致保存后的 profile 在重新加载时校验失败——一个只在"保存后再读取"
        路径上才暴露的 bug。
        """
        settings = Settings()
        settings.llm.max_tokens = 1234
        settings.agent.max_tokens = 5678
        save_named(settings, "tokencfg")

        raw = json.loads(settings_path("tokencfg").read_text(encoding="utf-8"))
        assert raw["llm"]["max_tokens"] == 1234
        assert raw["agent"]["max_tokens"] == 5678

        reloaded = load_settings(name="tokencfg", dotenv_path=None)
        assert reloaded.llm.max_tokens == 1234
        assert reloaded.agent.max_tokens == 5678

    def test_secret_field_names_detects_only_secret_str(self) -> None:
        from scitrace.config import LLMSettings
        from scitrace.config.settings import secret_field_names

        assert secret_field_names(LLMSettings) == ["api_key"]
        assert "max_tokens" not in secret_field_names(LLMSettings)


class TestEnvironmentParsing:
    def test_nested_delimiter(self) -> None:
        nested = env_to_nested({"SCITRACE_LLM__MODEL": "x", "SCITRACE_AGENT__MAX_STEPS": "3"})
        assert nested == {"llm": {"model": "x"}, "agent": {"max_steps": "3"}}

    def test_unrelated_variables_are_ignored(self) -> None:
        """进程环境里有大量无关变量，全盘接收会让 extra='forbid' 变成噪声源。"""
        assert env_to_nested({"PATH": "/usr/bin", "HOME": "/root"}) == {}

    def test_json_valued_variable_is_parsed(self) -> None:
        nested = env_to_nested({"SCITRACE_INGEST__SOURCE_DIRS": '["./a","./b"]'})
        assert nested["ingest"]["source_dirs"] == ["./a", "./b"]

    def test_malformed_json_value_falls_back_to_string(self) -> None:
        nested = env_to_nested({"SCITRACE_INGEST__SOURCE_DIRS": "[oops"})
        assert nested["ingest"]["source_dirs"] == "[oops"

    def test_list_env_var_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCITRACE_INGEST__SOURCE_DIRS", '["./papers"]')
        assert load_settings(dotenv_path=None).ingest.source_dirs == ["./papers"]

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("KEY=value", {"KEY": "value"}),
            ("KEY='quoted value'", {"KEY": "quoted value"}),
            ('KEY="dq value"', {"KEY": "dq value"}),
            ("export KEY=value", {"KEY": "value"}),
            ("# comment\nKEY=value", {"KEY": "value"}),
            ("\n\nKEY=value\n\n", {"KEY": "value"}),
            ("NOEQUALS", {}),
            ("KEY=a=b", {"KEY": "a=b"}),  # 值里的等号要保留（base64 key 常见）
            ("  KEY  =  value  ", {"KEY": "value"}),
        ],
    )
    def test_dotenv_parsing(self, tmp_path: Path, content: str, expected: dict) -> None:
        path = tmp_path / ".env"
        path.write_text(content, encoding="utf-8")
        assert read_dotenv(path) == expected

    def test_missing_dotenv_is_empty(self, tmp_path: Path) -> None:
        assert read_dotenv(tmp_path / "nope.env") == {}
        assert read_dotenv(None) == {}


class TestDeepMerge:
    def test_recursive_merge(self) -> None:
        assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}

    def test_lists_are_replaced_not_concatenated(self) -> None:
        """列表替换是刻意的：拼接会让用户无法通过覆盖来"缩小范围"。"""
        assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}

    def test_original_is_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestIndexFingerprint:
    def test_is_stable(self) -> None:
        assert Settings().index_fingerprint() == Settings().index_fingerprint()

    def test_is_12_hex(self) -> None:
        fingerprint = Settings().index_fingerprint()
        assert len(fingerprint) == 12
        assert all(char in "0123456789abcdef" for char in fingerprint)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda s: setattr(s.ingest, "parser", "pymupdf"), id="parser"),
            pytest.param(lambda s: setattr(s.ingest.chunking, "max_chars", 4000), id="chunking"),
            pytest.param(lambda s: setattr(s.embedding, "model", "other"), id="embedding_model"),
            pytest.param(lambda s: setattr(s.embedding, "backend", "openai"), id="embedding_backend"),
        ],
    )
    def test_changes_with_representation_config(self, mutate) -> None:
        """表示配置变化**必须**导致指纹变化，否则会静默复用不匹配的索引。"""
        before = Settings()
        after = Settings()
        mutate(after)
        assert before.index_fingerprint() != after.index_fingerprint()

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda s: setattr(s.ingest, "source_dirs", ["./papers"]), id="source_dirs"
            ),
            pytest.param(
                lambda s: setattr(s.retrieval, "strategy", "hybrid_rrf"), id="retrieval_strategy"
            ),
            pytest.param(lambda s: setattr(s.retrieval, "k", 99), id="retrieval_k"),
            pytest.param(lambda s: setattr(s.llm, "model", "other"), id="llm_model"),
        ],
    )
    def test_does_not_change_with_query_side_config(self, mutate) -> None:
        """只影响"怎么查"的配置不应导致索引重建。

        尤其 `source_dirs`：若它参与指纹，`stc index ./papers` 之后
        `stc ask`（不带路径）会因指纹不同而报"索引不存在"。见 SPEC §5.3 的修正说明。
        """
        before = Settings()
        after = Settings()
        mutate(after)
        assert before.index_fingerprint() == after.index_fingerprint()

    def test_schema_version_participates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = Settings().index_fingerprint()
        monkeypatch.setattr(settings_module, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
        assert Settings().index_fingerprint() != before

    def test_index_dir_uses_fingerprint(self) -> None:
        settings = Settings()
        assert settings.index_dir.name == settings.index_fingerprint()
        assert settings.sessions_dir.name == settings.index_fingerprint()


class TestModelValidation:
    def test_chunking_cross_field_constraints(self) -> None:
        """单看每个字段都合法、放在一起才矛盾的配置必须被拦住，

        否则症状表现为"分块结果诡异"而不是"配置错误"。
        """
        with pytest.raises(ValueError, match="min_chars"):
            ChunkingSettings(min_chars=2000, target_chars=1000, max_chars=3000)
        with pytest.raises(ValueError, match="target_chars"):
            ChunkingSettings(target_chars=5000, max_chars=3000)
        with pytest.raises(ValueError, match="overlap_chars"):
            ChunkingSettings(max_chars=1000, overlap_chars=1000)

    def test_llm_role_routing(self) -> None:
        settings = LLMSettings(model="strong", summary_model="cheap", agent_model="agentic")
        assert settings.model_for("main") == "strong"
        assert settings.model_for("summary") == "cheap"
        assert settings.model_for("agent") == "agentic"

    def test_llm_role_falls_back_to_main(self) -> None:
        settings = LLMSettings(model="only")
        assert settings.model_for("summary") == "only"
        assert settings.model_for("agent") == "only"

    def test_retrieval_strategy_literal(self) -> None:
        with pytest.raises(ValueError):
            RetrievalSettings(strategy="magic")  # type: ignore[arg-type]

    def test_reranker_flag(self) -> None:
        assert RetrievalSettings(strategy="hybrid_rrf_rerank").needs_reranker is True
        assert RetrievalSettings(strategy="hybrid_rrf").needs_reranker is False

    def test_invalid_env_value_reports_clearly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCITRACE_AGENT__MAX_STEPS", "not-a-number")
        with pytest.raises(SettingsError, match="配置校验失败"):
            load_settings(dotenv_path=None)

    def test_ingest_defaults(self) -> None:
        ingest = IngestSettings()
        assert ingest.parser == "pypdf"
        assert ".pdf" not in ingest.extra_suffixes  # pdf 由 parser 显式支持，不走后缀表

"""配置系统：分层加载、校验与索引指纹。

对外只需记住两个入口：

- :func:`load_settings` —— 按五层优先级装配配置（生产路径）；
- :class:`Settings` —— 配置模型本身，含 :meth:`Settings.index_fingerprint`。
"""

from scitrace.config.settings import (
    CONFIG_ROOT,
    DEFAULT_SETTINGS_NAME,
    ENV_PREFIX,
    AgentSettings,
    AnswerSettings,
    ChunkingSettings,
    EmbeddingSettings,
    IndexSettings,
    IngestSettings,
    LLMSettings,
    LLMRole,
    MetadataSettings,
    RetrievalSettings,
    ScreeningSettings,
    Settings,
    SettingsError,
    deep_merge,
    env_to_nested,
    load_settings,
    read_dotenv,
    read_named_payload,
    save_named,
    settings_path,
    sorted_available_profiles,
)

__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_SETTINGS_NAME",
    "ENV_PREFIX",
    "AgentSettings",
    "AnswerSettings",
    "ChunkingSettings",
    "EmbeddingSettings",
    "IndexSettings",
    "IngestSettings",
    "LLMSettings",
    "LLMRole",
    "MetadataSettings",
    "RetrievalSettings",
    "ScreeningSettings",
    "Settings",
    "SettingsError",
    "deep_merge",
    "env_to_nested",
    "load_settings",
    "read_dotenv",
    "read_named_payload",
    "save_named",
    "settings_path",
    "sorted_available_profiles",
]

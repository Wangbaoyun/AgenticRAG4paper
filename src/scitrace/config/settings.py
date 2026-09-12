"""分层配置：把"实验配置"变成一等公民。

## 为什么自己实现优先级合并，而不是全交给 pydantic-settings

``pydantic-settings`` 的默认优先级是 `init kwargs > 环境变量 > .env > 默认值`。
本 SPEC 需要的是**五层**（§5.1）：

    显式参数(CLI)  >  进程环境变量  >  命名配置文件  >  .env  >  内置默认

其中"命名配置文件"夹在中间——它比 `.env` 高（`.env` 是机器本地默认值），
又比进程环境变量低（CI 里用 `SCITRACE_LLM__MODEL=...` 临时覆盖一次实验，
不应该被仓库里的 profile 文件压过去）。

用框架的 `settings_customise_sources` 表达这一层需要向它注入"当前 profile 名"，
而 profile 名本身又可能来自环境变量或命令行，会形成循环依赖。
因此这里改为**显式合并**：把五层依次归并成一个嵌套字典，最后一次性交给 pydantic 校验。
代价是三十行合并代码；收益是优先级完全可读、可单测，且报错信息来自同一处校验。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from scitrace import SCHEMA_VERSION
from scitrace.util.hashing import sha256_hex, stable_json

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
    "load_settings",
    "read_dotenv",
    "read_named_payload",
    "save_named",
    "settings_path",
]

#: 配置根目录。所有本项目的本地状态（索引、会话、profile）都在其下，
#: 便于整体备份、迁移与清理，也避免污染用户的工作目录。
CONFIG_ROOT = Path.home() / ".scitrace"

ENV_PREFIX = "SCITRACE_"
ENV_NESTED_DELIMITER = "__"
DEFAULT_SETTINGS_NAME = "default"
DEFAULT_DOTENV = Path(".env")

LLMRole = Literal["main", "summary", "agent"]


class SettingsError(RuntimeError):
    """配置读取或校验失败。"""


class _ConfigGroup(BaseModel):
    """所有配置分组的基类。

    统一开启两项校验，缺一不可：

    - ``extra="forbid"``：拼错的字段名必须报错。否则 ``{"llm": {"modle": "x"}}``
      会安静地使用默认模型跑完实验——这是配置类错误里最难发现的一种。
    - ``validate_assignment=True``：**赋值时也校验**。CLI 与库内代码会就地修改配置
      （``settings.retrieval.k = -5``、``settings.retrieval.strategy = "typo"``），
      没有这一项，非法值会被原样带入运行期，症状要到很晚才浮现。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------- #
# 分组配置模型
# --------------------------------------------------------------------------- #


class LLMSettings(_ConfigGroup):
    """LLM 接入配置。

    三个角色可分别指定模型，这是**成本分级路由**（SPEC §9 增量 ④）的配置落点：
    证据筛选是"每篇论文调一次"的高频调用，答案合成是一次性的高质量调用，
    用一个强模型干两件事会显著推高成本。
    """

    model: str = Field(default="deepseek/deepseek-chat", description="主模型：答案合成")
    summary_model: str | None = Field(
        default=None, description="摘要模型：证据筛选。None 表示复用 model"
    )
    agent_model: str | None = Field(
        default=None, description="Agent 模型：工具选择。None 表示复用 model"
    )
    api_key: SecretStr | None = None
    api_base: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)

    def model_for(self, role: LLMRole) -> str:
        """返回某个角色应使用的模型标识。

        Note:
            ``role="agent"`` 未单独配置时会回落到 ``model``。若主模型不擅长
            工具调用（例如纯推理模型），Agent 循环会不稳定——此时请显式设置
            ``agent_model``，而不是依赖回落。
        """
        if role == "summary":
            return self.summary_model or self.model
        if role == "agent":
            return self.agent_model or self.model
        return self.model


class EmbeddingSettings(_ConfigGroup):
    """向量化配置。"""

    backend: Literal["local", "openai"] = "local"
    model: str = Field(default="BAAI/bge-small-zh-v1.5", description="参与索引指纹计算")
    api_key: SecretStr | None = None
    api_base: str | None = None
    batch_size: int = Field(default=32, ge=1)
    query_prefix: str = Field(
        default="",
        description="非对称编码模型的查询指令前缀（BGE 中文系列通常需要）",
    )
    dimension: int | None = Field(
        default=None, gt=0, description="期望维度；None 表示首次调用后自动探测"
    )


class ChunkingSettings(_ConfigGroup):
    """分块参数。修改其中任何一项都会改变索引指纹（SPEC §5.3）。"""

    target_chars: int = Field(default=1500, ge=100)
    max_chars: int = Field(default=3000, ge=200)
    min_chars: int = Field(default=300, ge=0)
    overlap_chars: int = Field(default=200, ge=0)
    drop_references: bool = True

    def model_post_init(self, __context: Any) -> None:
        """校验参数之间的相互关系。

        这些约束单看每个字段都合法，放在一起才矛盾；不校验的话，
        症状会表现为"分块结果诡异"而不是"配置错误"，排查成本极高。
        """
        if self.min_chars > self.target_chars:
            raise ValueError("chunking.min_chars 不能大于 target_chars")
        if self.target_chars > self.max_chars:
            raise ValueError("chunking.target_chars 不能大于 max_chars")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("chunking.overlap_chars 必须小于 max_chars，否则切分会不终止")


class IngestSettings(_ConfigGroup):
    """摄入配置。"""

    parser: str = Field(default="pypdf", description="PDF 解析后端名，参与索引指纹")
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    source_dirs: list[str] = Field(
        default_factory=list,
        description="默认语料路径。**不参与索引指纹**（SPEC §5.3）",
    )
    extra_suffixes: list[str] = Field(default_factory=lambda: [".txt", ".md"])
    max_file_mb: float = Field(default=200.0, gt=0.0)


class RetrievalSettings(_ConfigGroup):
    """检索策略配置——可插拔检索内核的开关（SPEC §3.6 / 增量 ②）。"""

    strategy: Literal["dense", "dense_mmr", "hybrid_rrf", "hybrid_rrf_rerank"] = "dense_mmr"
    k: int = Field(default=10, ge=1)
    fetch_k_multiplier: int = Field(default=4, ge=1, description="MMR 过采倍数")
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, description="RRF 平滑常数")
    rerank_pool_multiplier: int = Field(default=3, ge=1)
    rerank_model: str = "BAAI/bge-reranker-base"

    @property
    def needs_reranker(self) -> bool:
        """当前策略是否需要 Cross-Encoder 重排后端。"""
        return self.strategy == "hybrid_rrf_rerank"


class ScreeningSettings(_ConfigGroup):
    """证据筛选配置。"""

    backend: Literal["llm", "cross_encoder"] = "llm"
    min_relevance: int = Field(default=5, ge=0, le=10)
    concurrency: int = Field(default=8, ge=1, description="并发 LLM 调用数")
    presummary_n: int = Field(
        default=15, ge=1, description="cross_encoder 后端下，进入 LLM 摘要的片段数"
    )
    cross_encoder_model: str = "BAAI/bge-reranker-base"


class AnswerSettings(_ConfigGroup):
    """答案合成配置。"""

    max_evidence: int = Field(default=10, ge=1, description="进入上下文的证据条数上限")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description=(
            "答案合成的输出上限。**推理模型需要留出思考空间**——"
            "实测 deepseek-v4-flash 在本项目的合成提示词下会把 2048 全部用于思考，"
            "正文返回空串（finish_reason=length）。合成器有双倍预算重试兜底，"
            "但把默认值设在合理水平可以避免每次问答都白跑一轮。"
        ),
    )


class AgentSettings(_ConfigGroup):
    """Agent 运行时配置（SPEC §4）。"""

    max_steps: int = Field(default=12, ge=1)
    timeout_seconds: float = Field(default=500.0, gt=0.0)
    max_tokens: int | None = Field(
        default=None, ge=1, description="会话级 token 预算；None 表示不限"
    )
    max_cost_usd: float | None = Field(
        default=1.0, ge=0.0, description="会话级成本预算；None 表示不限"
    )
    context_token_limit: int = Field(default=32000, ge=1000)
    max_tool_param_retries: int = Field(default=5, ge=0, le=10)
    no_new_evidence_limit: int = Field(
        default=2, ge=1, description="连续无新增证据达到该次数则自动收尾"
    )


class MetadataSettings(_ConfigGroup):
    """元数据补全配置。"""

    enabled: bool = True
    providers: list[str] = Field(
        default_factory=lambda: ["crossref", "semantic_scholar", "openalex"],
        description="按**优先级从高到低**排列；合并时高优先级优先",
    )
    crossref_mailto: str | None = Field(
        default=None, description="Crossref 建议提供，可显著提高限流额度"
    )
    semantic_scholar_api_key: SecretStr | None = None
    openalex_mailto: str | None = None
    retraction_csv: Path | None = None
    llm_title_inference: bool = Field(
        default=False,
        description=(
            "本地与在线元数据都拿不到标题时，是否用 LLM 从首页推断一个。"
            "**默认关闭**：开启后索引将从「纯本地计算」变成「依赖外部服务并产生 token 成本」，"
            "这个变化必须由使用者显式选择。"
        ),
    )
    llm_title_max_chars: int = Field(
        default=2000, ge=200, description="送进标题推断的首页字符数上限"
    )
    timeout_s: float = Field(default=15.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrency: int = Field(default=4, ge=1)


class IndexSettings(_ConfigGroup):
    """索引与会话存储位置。"""

    root: Path = CONFIG_ROOT / "index"
    sessions_root: Path = CONFIG_ROOT / "sessions"
    cache_embeddings: bool = True


class Settings(BaseSettings):
    """全部配置的聚合根。

    通常经 :func:`load_settings` 构造（它实现五层优先级）。
    直接 ``Settings()`` 也可用，此时只有"进程环境变量 + 默认值"两层，
    适合单元测试与库内调用。
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        extra="forbid",
        # 赋值时也校验：`settings.retrieval.k = -5` 这类代码层面的笔误应当立刻报错，
        # 而不是带着非法值跑完整个实验。
        validate_assignment=True,
        # 刻意不在此处声明 env_file：.env 由 load_settings 显式加载，
        # 以保证优先级链在一处可见、可测。
        env_file=None,
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    metadata: MetadataSettings = Field(default_factory=MetadataSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)

    # ---------------------------------------------------------------- 派生值 --

    def index_fingerprint(self) -> str:
        """计算索引指纹（SPEC §5.3）。

        只纳入**表示配置**：解析器、分块参数、嵌入后端与模型。
        刻意不含语料路径、检索策略、提示词——前者的理由见 SPEC §5.3 的修正说明；
        后两者只影响"怎么查"，不影响"文本被存成什么样"，纳入会导致
        改一次检索策略就白建一遍索引。
        """
        payload = {
            "schema": SCHEMA_VERSION,
            "parser": self.ingest.parser,
            "chunking": self.ingest.chunking.model_dump(),
            "embedding_backend": self.embedding.backend,
            "embedding_model": self.embedding.model,
        }
        return sha256_hex(stable_json(payload), length=12)

    @property
    def index_dir(self) -> Path:
        """当前配置对应的索引目录。"""
        return Path(self.index.root) / self.index_fingerprint()

    @property
    def sessions_dir(self) -> Path:
        """当前配置对应的会话记录目录。"""
        return Path(self.index.sessions_root) / self.index_fingerprint()

    # ------------------------------------------------------------ 序列化 --

    def to_payload(self) -> dict[str, Any]:
        """导出为可写入 profile 文件的嵌套字典。

        **永不包含密钥。** 命名 profile 的设计用途是被提交进仓库以复现实验
        （README 与 SPEC §5.1 均如此建议），把 API key 写进去等于埋雷；
        密钥应放在 ``.env`` 或进程环境变量里——加载器原生支持这两层。
        因此这里不提供"包含密钥"的开关：一个会被误用的开关不如没有。

        遮罩依据**字段类型**（``SecretStr``）而非字段名。按名字匹配
        （"含 key/token 的字段"）会误伤 ``llm.max_tokens`` / ``agent.max_tokens``
        这类正常字段，把整数置成 ``None`` 从而让 profile 在重新加载时校验失败。
        """
        payload = self.model_dump(mode="json")
        for group_name in ("llm", "embedding", "metadata"):
            group = getattr(self, group_name)
            for field_name in secret_field_names(type(group)):
                payload[group_name][field_name] = None
        return payload


# --------------------------------------------------------------------------- #
# 五层优先级加载
# --------------------------------------------------------------------------- #


def secret_field_names(model_type: type[BaseModel]) -> list[str]:
    """返回模型中类型为 :class:`SecretStr` 的字段名。

    按**类型**而非字段名识别密钥，使 ``max_tokens`` 这类名字里带 "token"
    的普通字段不会被误判。
    """
    return [
        name
        for name, field in model_type.model_fields.items()
        if field.annotation is SecretStr or SecretStr in get_args(field.annotation)
    ]


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并两个嵌套字典，``overlay`` 覆盖 ``base``。

    只对**两侧都是字典**的键递归；其余一律替换——包括列表。
    列表替换（而非拼接）是刻意的：``source_dirs``、``providers`` 这类配置
    如果拼接，用户就无法通过覆盖来"缩小范围"，而缩小范围是常见需求。
    """
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = value
    return result


def read_dotenv(path: Path | None) -> dict[str, str]:
    """读取 ``.env`` 文件为扁平字典。

    只支持实际会用到的最小子集：``KEY=VALUE``、``#`` 注释、``export`` 前缀、
    成对引号包裹的值。刻意不实现变量插值与多行值——需要这些能力时应当
    直接使用环境变量，而不是把一个 .env 解析器养成一个小项目。
    """
    if path is None or not Path(path).is_file():
        return {}
    result: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def _coerce_env_value(raw: str) -> Any:
    """把环境变量字符串还原为结构化值。

    ``SCITRACE_INGEST__SOURCE_DIRS=["./a","./b"]`` 这类列表/对象配置
    必须以 JSON 书写；其余按普通字符串处理，交由 pydantic 做类型转换。
    """
    stripped = raw.strip()
    if stripped[:1] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw
    return raw


def env_to_nested(flat: Mapping[str, str]) -> dict[str, Any]:
    """把 ``SCITRACE_A__B=x`` 形式的扁平环境变量转换为嵌套字典。

    非 ``SCITRACE_`` 前缀的变量被忽略——进程环境里有大量无关变量，
    全盘接收只会让 `extra="forbid"` 变成噪声源。
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [part for part in key[len(ENV_PREFIX) :].split(ENV_NESTED_DELIMITER) if part]
        if not path:
            continue
        cursor = nested
        for part in path[:-1]:
            child = cursor.setdefault(part.lower(), {})
            if not isinstance(child, dict):  # 前缀冲突，如同时出现 A 与 A__B
                break
            cursor = child
        else:
            cursor[path[-1].lower()] = _coerce_env_value(value)
    return nested


def settings_path(name: str = DEFAULT_SETTINGS_NAME) -> Path:
    """返回命名 profile 的文件路径。"""
    return CONFIG_ROOT / "settings" / f"{name}.json"


def read_named_payload(name: str | None) -> dict[str, Any]:
    """读取命名 profile。

    Raises:
        SettingsError: 显式指定的 profile 不存在或格式非法。
            刻意**不**静默回落到默认值——``--settings my_deepseek`` 打错一个字母
            就用默认配置跑完整个实验，是最难发现的一类错误。
    """
    profile_name = name or DEFAULT_SETTINGS_NAME
    path = settings_path(profile_name)
    if not path.is_file():
        if name and name != DEFAULT_SETTINGS_NAME:
            raise SettingsError(
                f"配置文件不存在：{path}\n可用 profile：{', '.join(sorted_available_profiles()) or '(无)'}"
            )
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SettingsError(f"配置文件不是合法 JSON：{path}：{error}") from error
    if not isinstance(payload, dict):
        raise SettingsError(f"配置文件顶层必须是对象：{path}")
    return payload


def sorted_available_profiles() -> list[str]:
    """列出已有的 profile 名。"""
    directory = CONFIG_ROOT / "settings"
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def load_settings(
    *,
    name: str | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """按五层优先级加载配置（SPEC §5.1）。

    优先级由低到高::

        内置默认  <  .env 文件  <  命名 profile  <  进程环境变量  <  显式覆盖(CLI)

    Args:
        name: 命名 profile 名；``None`` 表示 ``default``。
        dotenv_path: ``.env`` 路径；``None`` 表示不读取。
        overrides: 最高优先级的覆盖（通常来自命令行参数），嵌套字典形式。

    Returns:
        校验完成的 :class:`Settings`。

    Raises:
        SettingsError: profile 缺失/非法，或最终配置未通过校验。
    """
    layers: dict[str, Any] = {}
    layers = deep_merge(layers, env_to_nested(read_dotenv(dotenv_path)))
    layers = deep_merge(layers, read_named_payload(name))
    layers = deep_merge(
        layers,
        env_to_nested({key: value for key, value in os.environ.items() if key.startswith(ENV_PREFIX)}),
    )
    if overrides:
        layers = deep_merge(layers, overrides)

    try:
        return Settings(**layers)
    except Exception as error:  # noqa: BLE001 — 统一转为本项目的异常类型
        raise SettingsError(f"配置校验失败：{error}") from error


def save_named(settings: Settings, name: str = DEFAULT_SETTINGS_NAME) -> Path:
    """把配置保存为命名 profile（不含密钥）。

    Returns:
        写入的文件路径。
    """
    path = settings_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.to_payload()
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path

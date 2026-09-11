#!/usr/bin/env python3
"""洁净室（clean-room）重写项目的反抄袭 / 相似度审计脚本。

本脚本用于回答一个非常具体的问题：**新项目（target）的源码文本、Prompt 字符串、
注释与文档中，是否存在来自上游参考实现（reference）的受著作权保护的"表达"？**

设计原则
--------
1. 只读上游：参考侧文件集一律通过 ``git ls-files`` + ``git show HEAD:<path>`` 获取，
   因此比对基准永远是 **HEAD 中 git 跟踪的干净版本**，不受工作区改动影响，
   也不会把上游仓库里用户自己新增的未跟踪文件（例如 ``src/paperqa/types_notes.py``、
   ``docs/*.md``、``my_papers/``）当成"上游代码"。
2. 只使用 Python 标准库（3.11+），无任何第三方依赖。
3. 每项检查独立计分、独立判定通过/失败；任一项失败则进程退出码为 1。
4. 本脚本自身（及其测试）必然包含被禁词表，因此审计时会把这些"审计工具自身的文件"
   从「禁用标识符」「依赖审计」两项中排除（详见 ``is_audit_tooling``）。

检查项
------
1. token n-gram 重叠率（默认 n=8，保留注释与 docstring）
2. 逐行精确复制检测（忽略 <12 字符的行）
3. 字符串字面量相似度（默认阈值 0.6）+ 上游专有标识符专项检查
4. 依赖审计（pyproject.toml / requirements*.txt / import 语句）
5. 资产与文档审计（SHA-256 内容哈希 + .md 的 n=12 token n-gram）

退出码
------
* 0：全部检查通过
* 1：存在失败检查项
* 2：参数 / 路径 / git 环境错误

用法::

    python tools/similarity_audit.py --target /home/wby/MyPaperQA \\
        --reference /home/wby/PaperQA --report docs/AUDIT.md \\
        [--ngram 8] [--fail-threshold 0.02]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import keyword
import math
import bisect
import difflib
import fnmatch
import hashlib
import io
import re
import subprocess
import sys
import tokenize
import tomllib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

# --------------------------------------------------------------------------- #
# 默认阈值与常量
# --------------------------------------------------------------------------- #

DEFAULT_NGRAM: int = 8
DEFAULT_DOC_NGRAM: int = 12
DEFAULT_FAIL_THRESHOLD: float = 0.02
DEFAULT_FILE_THRESHOLD: float = 0.05
DEFAULT_STRING_THRESHOLD: float = 0.6
DEFAULT_TOP_N: int = 20
DEFAULT_MAX_COPIED_LINES: int = 0
DEFAULT_MIN_COPIED_BLOCK: int = 5
DEFAULT_MIN_MATCHED_CHARS: int = 32

#: 小于该 n-gram 总数的文件在单文件 containment 判定上不稳定。
#: 这类文件**不参与单文件判定**（其共享 n-gram 仍会列出供人工核验）：
#: 一个 94 个 n-gram 的小模块，只要十来个通用 token 序列重合就会算出 13% 的
#: containment——这是小样本方差，不是复制证据。首轮审计即被此效应误报。
SMALL_FILE_NGRAMS: int = 300

#: 标识符切分（用于文档频率统计与区分性判据）。
IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")

#: 完整形如 URL 的字符串。第三方服务的 API 端点由服务方规定，
#: 使用该服务就必须使用它的 URL——属于事实性标识符，不含作者表达。
#: 只豁免**整个字符串就是一个 URL** 的情形：含 URL 的长文本仍走常规比对。
URL_LITERAL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)

#: 字面量字符串切分（粗略但足够：单双引号成对）。
STRING_LITERAL_RE = re.compile(r"\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*'")

#: 判定"通用标识符"的**文档频率**阈值。
#:
#: 在超过该比例的上游文件中出现过的标识符，视为 Python 生态的通用词汇
#: （``self`` 53%、``len`` 64%、``response`` 40%、``settings`` 36%、``result`` 31%…），
#: 它们由语言习惯或领域通用语决定，不承载作者的命名选择。
#:
#: 这是把逐行检查从"白名单"改为"区分性内容"判据的关键：
#: 白名单方法不收敛——Python 的规范写法是开放集合，每轮审计都能找到新的
#: "显然通用但未被规则覆盖"的行（本项目实测连续三轮各 4/22/15 行）。
#: 文档频率是信息检索里的标准做法（MOSS/JPlag 一类系统用它压制趋同噪声），
#: 且它**直接指向真正的信号**：复制会带来上游的**特征标识符**
#: （``citation_regex`` 1.7%、``ptext`` 1.7%、``similarity_search`` 3.4%）。
#:
#: 阈值 10% 由实测确定：本项目全部误报行的标识符 DF ≥ 22%，
#: 而植入复制样本的标识符 DF ≤ 7%，两者之间有很宽的安全间隔。
COMMON_IDENTIFIER_DF: float = 0.10

#: 独立性守卫工具的自声明标记。文件前 20 行含此标记者，
#: 从「依赖审计」「专有标识符」两项中豁免（理由见 :func:`is_audit_tooling`）。
AUDIT_TOOLING_MARKER: str = "scitrace: independence-guard"

#: 判定"连续搬运"所需的**最短公共 token 连续片段长度**。
#: 与 n-gram containment 组成双重条件：只有统计超标且存在这么长的连续公共片段，
#: 才判为失败。阈值取 25 的依据是实测——两个使用同一框架（pydantic）、
#: 同一领域（学术文献）的项目，脚手架序列的随机重合通常止步于十余个 token，
#: 而真正的整段搬运很容易达到几十个。
MIN_COMMON_TOKEN_RUN: int = 25

#: 探测最长公共片段时使用的长度阶梯（从大到小逐级探测，命中即返回）。
COMMON_RUN_LADDER: tuple[int, ...] = (100, 60, 40, 25)

MIN_LINE_LENGTH: int = 12
MIN_STRING_LENGTH: int = 20
MAX_TEXT_BYTES: int = 5 * 1024 * 1024
PREVIEW_CHARS: int = 140

#: 目标侧遍历时跳过的目录名（任意层级）。
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "env",
        "node_modules",
        "build",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        ".idea",
        ".vscode",
        ".DS_Store",
    }
)

#: 目录名后缀命中即跳过（如 ``scitrace.egg-info``）。
SKIP_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)

#: 会做行级 / n-gram 源码比对的扩展名。
PY_SUFFIXES: frozenset[str] = frozenset({".py", ".pyi"})

#: 会做"文档 n-gram"比对的扩展名。
DOC_SUFFIXES: frozenset[str] = frozenset({".md", ".rst"})

#: 逐行比对的三个层级。
TIER_BOILERPLATE = "样板"
TIER_GENERIC = "通用惯用式"
TIER_SUBSTANTIVE = "实质性"

#: 「上游专有标识符」检查会扫描的代码 / 配置扩展名。
CODE_SCAN_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".toml",
        ".cfg",
        ".ini",
        ".txt",
        ".json",
        ".json5",
        ".yaml",
        ".yml",
        ".jinja",
        ".jinja2",
        ".j2",
        ".sh",
        ".sql",
    }
)

#: 依赖禁用清单：规范化后的包名 -> 说明。命中即失败。
BANNED_DEPENDENCIES: dict[str, str] = {
    "paperqa": "上游核心发行包（paper-qa / paperqa / paper_qa）",
    "fhaviary": "上游同组织（Future-House）的配套包",
    "aviary": "上游同组织的配套包",
    "fhlmi": "上游同组织的 LLM 抽象包",
    "ldp": "上游同组织的文档解析包",
}

#: 只警告不判失败的依赖（通用 LLM 抽象层，可独立使用）。
WARN_DEPENDENCIES: dict[str, str] = {
    "lmi": "上游同组织维护的通用 LLM 抽象层（可选用，仅警告）",
}

#: 上游专有标识符（大小写不敏感的子串匹配，命中即违规）。
PROHIBITED_IDENTIFIERS: tuple[str, ...] = (
    "pqac",
    "paperqa",
    "paper-qa",
    "paper_qa",
    "pqa",
)

#: 行级比对中被视为"通用样板"的白名单（不具独创性，因此完全不参与比对）。
BOILERPLATE_LINE_RE = re.compile(
    r"^(?:"
    r"from\s+[\w.]+\s+import\s+.+|"
    r"import\s+[\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*|"
    r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:|"
    r"sys\.exit\(\w*\(\s*\)\)"
    r")$"
)

#: 行级比对中的"通用惯用式"：会被计数展示，但**不单独构成失败**
#: （它们是与 <12 字符规则同性质的"必然重复"，例如类型守卫、异常捕获、关键字实参行）。
GENERIC_IDIOM_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^if\s+(?:not\s+)?isinstance\([^()]*\)\s*:\s*$"),
    re.compile(r"^except\b[\w\s,().]*(?:\bas\s+\w+)?\s*:\s*$"),
    re.compile(
        r"^return\s*(?:[A-Za-z_][\w.]*|None|True|False|\d+|[\"'][^\"']*[\"']|\(\)|\[\]|\{\})?\s*$"
    ),
    re.compile(r"^yield\s*(?:[A-Za-z_][\w.]*|None|\d+)?\s*$"),
    re.compile(r"^raise\s+[A-Za-z_][\w.]*(?:\([^()]*\))?\s*$"),
    re.compile(r"^[A-Za-z_]\w*\s*=\s*[^=].*,\s*$"),
    re.compile(r"^[A-Za-z_][\w.]*\s*\(\s*$"),
    re.compile(r"^[)\]}]+[,:]?\s*$"),
    re.compile(r"^([A-Za-z_]\w*)\s*=\s*\1\.[\w.]*\([^()]*\)\s*$"),
    re.compile(r"^(?:async\s+)?(?:with|for)\b[^:=]{0,50}:\s*$"),
    re.compile(r"^(?:else|try|finally)\s*:\s*$"),
    re.compile(r"^@[\w.]+(?:\([^()]*\))?\s*$"),
    re.compile(r"^(?:pass|break|continue|\.\.\.)\s*$"),
    # 列表 / __all__ 中的裸名称元素
    re.compile(r"^[A-Za-z_][\w.]*,\s*$"),
    # 无参构造赋值：settings = Settings()
    re.compile(r"^[A-Za-z_]\w*\s*=\s*[A-Za-z_][\w.]*\(\s*\)\s*$"),
    # 以未闭合的调用括号结尾的行（调用头，参数在后续行）
    re.compile(r"^.*\(\s*$"),
    # 类声明头：class Foo(BaseModel): —— 声明本身不承载表达
    re.compile(
        r"^class\s+\w+\(\s*(?:BaseModel|BaseSettings|Enum|StrEnum|IntEnum|Protocol|ABC|"
        r"TypedDict|NamedTuple|Generic\[[^]]*\]|object)?\s*\)\s*:\s*$"
    ),
    re.compile(r"^class\s+\w+\s*:\s*$"),
    # 字段声明：name: type 或 name: type = <简单字面量 / Field(...)>
    re.compile(
        r"^[A-Za-z_]\w*\s*:\s*[\w\[\]|, .\"']+"
        r"(?:=\s*(?:None|True|False|\d+|[\"'][^\"']*[\"']|Field\([^()]*\)|ConfigDict\([^()]*\)))?,?\s*$"
    ),
    # 简单布尔守卫：if not x: / if x is None: / if x is not None:
    re.compile(r"^if\s+not\s+[A-Za-z_][\w.]*(?:\([^()]*\))?\s*:\s*$"),
    re.compile(r"^if\s+[A-Za-z_][\w.\[\]\"']*\s+is(?:\s+not)?\s+None\s*:\s*$"),
    # 容器 / 配置构造赋值：model_config = ConfigDict(extra="forbid")
    re.compile(
        r"^[A-Za-z_]\w*\s*=\s*(?:ConfigDict|Field|FieldInfo|dict|list|set|tuple|str|int|float|bool)"
        r"\([^()]*\)\s*$"
    ),
    # Python 数据模型钩子：`def __len__(self) -> int:` 这类签名由语言强制规定
    # （数据模型协议确定了名字、参数与返回类型），作者没有任何表达空间。
    # 法理上属于"表达与思想合并"（merger doctrine）：表达方式被功能唯一决定时不受保护。
    # 本规则有实证依据：首轮审计把 4 行 `__len__` / `clear` 存根误判为"实质性复制"。
    re.compile(
        r"^(?:async\s+)?def\s+__\w+__\s*\([^)]*\)\s*"
        r"(?:->\s*[\w.\[\]|, ]+)?\s*:\s*$"
    ),
    # Protocol / ABC 存根签名：只有 self，返回类型是内置简单类型。
    # 接口方法的**名字与签名由契约决定**，实现方无选择余地，故同样不构成表达。
    re.compile(
        r"^(?:async\s+)?def\s+[a-z_]\w*\s*\(\s*self\s*\)\s*"
        r"->\s*(?:None|bool|int|float|str|bytes)(?:\s*\|\s*None)?\s*:\s*$"
    ),
    # ---- 语言 / 第三方 API 强制规定形式的行 ----
    #
    # 法理同"表达与思想合并"（merger doctrine）：当完成某个功能只有一种写法时，
    # 这种写法不受保护。下列各类都由语言规范或库的 API 形式唯一确定，
    # **作者没有任何选择余地**，因此不可能构成"表达"上的复制。
    #
    # 引入依据：第二轮审计在适配器层报出 22 行"实质性复制"，逐行核查后全部属于
    # 下列各类——其中 `logger = logging.getLogger(__name__)` 一项就占 9 行
    # （Python 生态该功能的标准写法，事实上不存在第二种）。
    #
    # 安全性由**判据分层**保证：本组规则只影响逐行检查（辅助判据），
    # 决定性的"≥25 token 连续公共片段"判据不受任何豁免影响。
    #
    # 模块日志器：标准写法
    re.compile(r"^logger\s*=\s*logging\.getLogger\(__name__\)$"),
    # 泛型变量声明：T = TypeVar("T") / T = TypeVar("T", bound=BaseModel)
    re.compile(r"^[A-Z]\w*\s*=\s*TypeVar\(\s*[\"']\w+[\"'](?:\s*,\s*bound=[^)]*)?\s*\)$"),
    # **无参数**的 API 调用语句：writer.commit() / await client.aclose() /
    # response.raise_for_status()。限定为无参调用——带参数的调用会体现作者的选择。
    re.compile(r"^(?:await\s+)?[\w.]+\(\)$"),
    # 工具 / 消息 schema 中的固定字段名与枚举值
    re.compile(
        r"^[\"'](?:type|role|name|content|id|function|arguments|tool_calls|index|embedding)[\"']"
        r"\s*:\s*[\"'][\w-]+[\"'],?$"
    ),
    # 测试断言的最简形式：assert results
    re.compile(r"^assert\s+[A-Za-z_][\w.\[\]]*$"),
    # 无函数体的裸比较守卫：if fetch_k < k:
    re.compile(r"^if\s+[\w.\[\]]+\s*(?:[<>]=?|==|!=)\s*[\w.\[\]]+\s*:\s*$"),
    # 目录创建的规范写法：关键字参数由 pathlib 文档规定，不存在第二种合理写法
    re.compile(r"^[\w.]+\.mkdir\(\s*parents=True\s*,\s*exist_ok=True\s*\)$"),
)

# --------------------------------------------------------------------------- #
# 分词器
# --------------------------------------------------------------------------- #

#: Python 源码分词器。刻意 **保留注释与字符串字面量**（它们同样是受保护的表达）。
PY_TOKEN_RE = re.compile(
    r"""
      (?P<comment>\#[^\n]*)
    | (?P<string>(?i:[rubf]{0,2})(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"))
    | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?)
    | (?P<op>[^\sA-Za-z0-9_#\"']+)
    | (?P<other>[^\s])
    """,
    re.VERBOSE,
)

#: 散文 / Markdown 分词器：词、数字、以及单个标点或单字（含中文）。
TEXT_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|\d+(?:\.\d+)*|[^\sA-Za-z0-9_]")

#: 字符串字面量回退提取（ast 解析失败时使用）。
FALLBACK_STRING_RE = re.compile(
    r"(?i:[rubf]{0,2})(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:\\.|[^'\\\n])*'|\"(?:\\.|[^\"\\\n])*\")"
)

#: 依赖名规范化（PEP 503）。
DEP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


class AuditError(RuntimeError):
    """参数 / 路径 / git 环境错误（对应退出码 2）。"""


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefFile:
    """上游 HEAD 中的一个 git 跟踪文件。"""

    rel: str
    data: bytes
    sha256: str
    text: str | None  # None 表示二进制或过大，不做文本分析


@dataclass(frozen=True)
class TargetFile:
    """目标项目中的一个文件。"""

    rel: str
    abspath: Path
    data: bytes
    text: str | None


@dataclass
class CheckResult:
    """单项检查结果。"""

    key: str
    title: str
    passed: bool
    metric: str
    threshold: str
    details: list[str] = field(default_factory=list)


@dataclass
class AuditResult:
    """一次完整审计的结果。"""

    target: Path
    reference: Path
    head_commit: str
    head_subject: str
    ngram: int
    doc_ngram: int
    fail_threshold: float
    file_threshold: float
    string_threshold: float
    min_matched_chars: int
    max_copied_lines: int
    min_copied_block: int
    generated_at: str
    reference_files: int
    reference_skipped: int
    reference_source: str
    target_files: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #


def _iso_now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """把任意文本压成单行预览。"""
    flat = text.replace("\r\n", "\n").replace("\n", "\\n").replace("\t", "\\t")
    if len(flat) > limit:
        return flat[:limit] + "…"
    return flat


def _md_code(text: str) -> str:
    """把文本放进 Markdown 行内代码块，自动选择合适长度的反引号栅栏。"""
    flat = _preview(text).replace("|", "\\|")
    longest = 0
    current = 0
    for char in flat:
        current = current + 1 if char == "`" else 0
        longest = max(longest, current)
    fence = "`" * (longest + 1)
    pad = " " if flat.startswith("`") or flat.endswith("`") else ""
    return f"{fence}{pad}{flat}{pad}{fence}"


def _md_cell(text: str) -> str:
    return _preview(text).replace("|", "\\|")


def _md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _normalize_dep_name(raw: str) -> str:
    """PEP 503 规范化：小写 + 连续 ``-_.`` 折叠为 ``-``。"""
    return re.sub(r"[-_.]+", "-", raw.strip().lower())


def _dep_hits(name: str, table: dict[str, str]) -> str | None:
    """返回命中的禁用清单条目说明；未命中返回 None。

    ``paperqa`` / ``paper-qa`` / ``paper_qa`` / ``paper_qa_docling`` 在 PEP 503 下
    并不完全等价，因此这里同时比较"规范化形式"与"去掉分隔符的紧凑形式"。
    前缀匹配只对长度 ≥ 5 的禁用名生效，避免 ``ldp`` 误伤 ``ldpc`` 之类的合法包。
    """
    raw = name.strip().lower()
    compact = re.sub(r"[-_.]+", "", raw)
    normalized = _normalize_dep_name(raw)
    for banned, reason in table.items():
        banned_compact = re.sub(r"[-_.]+", "", banned.lower())
        banned_normalized = _normalize_dep_name(banned)
        if compact == banned_compact or normalized == banned_normalized:
            return reason
        if len(banned_compact) >= 5 and compact.startswith(banned_compact):
            return reason
        if normalized.startswith(banned_normalized + "-"):
            return reason
    return None


def is_audit_tooling(rel_posix: str, text: str | None = None) -> bool:
    """判断目标侧文件是否属于"审计与独立性守卫工具自身"。

    这些文件的工作就是**定义与检测**禁用词（``paperqa``、``pqac``…），
    因此必然包含它们，还会包含用于构造反例的字符串。若不排除，审计会指控自己。

    判定依据有两条，都刻意保持狭窄且可在报告中核对：

    1. 文件名含 ``similarity_audit``，或为 ``tools/README.md``；
    2. 文件**前 20 行**含显式自声明标记 :data:`AUDIT_TOOLING_MARKER`。
       标记必须出现在文件头部，一眼可见、无法藏在中段；使用该标记的文件会在
       报告中单独列出，便于人工复核豁免是否正当。

    Args:
        rel_posix: 相对路径（POSIX 分隔符）。
        text: 文件文本；为 ``None``（二进制或超大文件）时只按文件名判定。
    """
    name = PurePosixPath(rel_posix).name
    if "similarity_audit" in name or rel_posix == "tools/README.md":
        return True
    if text:
        head = "\n".join(text.splitlines()[:20])
        if AUDIT_TOOLING_MARKER in head:
            return True
    return False


# --------------------------------------------------------------------------- #
# 目标侧 / 参考侧文件收集
# --------------------------------------------------------------------------- #


def _should_skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in SKIP_DIR_SUFFIXES)


def _decode_text(data: bytes) -> str | None:
    """二进制或超大文件返回 None。"""
    if len(data) > MAX_TEXT_BYTES:
        return None
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def iter_target_files(
    target: Path,
    *,
    report_path: Path | None,
    excludes: Sequence[str],
) -> list[TargetFile]:
    """遍历目标目录，跳过 VCS / 缓存 / 构建目录、审计报告自身与 ``--exclude`` 模式。"""
    report_resolved = report_path.resolve() if report_path is not None else None
    files: list[TargetFile] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel_parts = path.relative_to(target).parts
        if any(_should_skip_dir(part) for part in rel_parts[:-1]):
            continue
        rel = PurePosixPath(*rel_parts).as_posix()
        if report_resolved is not None and path.resolve() == report_resolved:
            continue
        if any(
            fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel_parts[-1], pattern)
            for pattern in excludes
        ):
            continue
        data = path.read_bytes()
        files.append(TargetFile(rel=rel, abspath=path, data=data, text=_decode_text(data)))
    return files


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - 环境缺 git
        raise AuditError("找不到 git 可执行文件，无法读取上游 HEAD 快照") from exc


def git_head_commit(repo: Path) -> str:
    proc = _run_git(repo, "rev-parse", "HEAD")
    if proc.returncode != 0:
        raise AuditError(
            f"`git -C {repo} rev-parse HEAD` 失败：{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout.decode("utf-8", "replace").strip()


def git_head_subject(repo: Path) -> str:
    proc = _run_git(repo, "log", "-1", "--pretty=%s")
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", "replace").strip()


def git_tracked_paths(repo: Path) -> tuple[list[str], str]:
    """返回（HEAD 跟踪的路径列表, 来源说明）。

    首选 ``git ls-files``（索引 = 用户实际检出的跟踪文件集）；若索引为空但 HEAD 有内容，
    则回退到 ``git ls-tree -r --name-only HEAD``，并在报告中注明来源。
    """
    proc = _run_git(repo, "-c", "core.quotePath=false", "ls-files", "-z")
    if proc.returncode != 0:
        raise AuditError(
            f"`git -C {repo} ls-files` 失败：{proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    paths = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
    if paths:
        return paths, "git ls-files（索引中的跟踪文件）"
    fallback = _run_git(repo, "-c", "core.quotePath=false", "ls-tree", "-r", "--name-only", "-z", "HEAD")
    if fallback.returncode != 0:
        raise AuditError(
            f"`git -C {repo} ls-tree HEAD` 失败：{fallback.stderr.decode('utf-8', 'replace').strip()}"
        )
    paths = [p for p in fallback.stdout.decode("utf-8", "replace").split("\0") if p]
    return paths, "git ls-tree -r HEAD（索引为空时的回退）"


def load_reference_files(repo: Path) -> tuple[list[RefFile], int, str]:
    """读取上游 HEAD 中每个跟踪文件的原始内容（``git show HEAD:<path>``）。

    在 HEAD 中不存在的路径会被跳过（例如索引里有、但从未提交的文件）。
    """
    paths, source = git_tracked_paths(repo)
    files: list[RefFile] = []
    skipped = 0
    for rel in paths:
        proc = _run_git(repo, "show", f"HEAD:{rel}")
        if proc.returncode != 0:
            skipped += 1
            continue
        data = proc.stdout
        files.append(
            RefFile(
                rel=rel,
                data=data,
                sha256=hashlib.sha256(data).hexdigest(),
                text=_decode_text(data),
            )
        )
    return files, skipped, source


# --------------------------------------------------------------------------- #
# 分词与 n-gram
# --------------------------------------------------------------------------- #


def tokenize_python(text: str) -> list[str]:
    return [match.group(0) for match in PY_TOKEN_RE.finditer(text)]


def tokenize_text(text: str) -> list[str]:
    return [match.group(0) for match in TEXT_TOKEN_RE.finditer(text)]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def containment(target: set[Any], reference: set[Any]) -> float:
    if not target:
        return 0.0
    return len(target & reference) / len(target)


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# --------------------------------------------------------------------------- #
# 字符串字面量提取
# --------------------------------------------------------------------------- #


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _fallback_strings(text: str) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for match in FALLBACK_STRING_RE.finditer(text):
        raw = match.group(0)
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        if isinstance(value, str):
            line = text.count("\n", 0, match.start()) + 1
            results.append((value, line))
    return results


def extract_string_literals(text: str) -> tuple[list[tuple[str, int]], bool]:
    """提取非 docstring 的字符串字面量。

    返回 ``( [(value, lineno), ...], used_ast )``；ast 解析失败时回退到正则提取，
    此时 docstring 无法识别（调用方会在报告中注明）。
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return _fallback_strings(text), False

    docstrings = _docstring_node_ids(tree)
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            results.append((node.value, getattr(node, "lineno", 0)))
    results.sort(key=lambda item: item[1])
    return results, True


@dataclass(frozen=True)
class StringPair:
    """一条"目标串 ↔ 最相似上游串"的比对结果。"""

    ratio: float
    matched: int
    target: tuple[str, str, int]
    reference: tuple[str, str, int]
    violation: bool


def normalize_for_comparison(value: str) -> str:
    """把结构化输出串归一化为"只比较内容，不比较骨架"的形式。

    **为什么需要这一步**：本项目与参考实现都会要求模型返回
    ``{"summary": ..., "relevance_score": N}`` 这样的结构化结果，因此
    测试夹具里必然出现大量同形的 JSON 骨架。原始字符串比对会把这种
    **由接口定义决定的骨架**算成高相似度，而那些字段名是本项目自己的
    提示词与规格定义的（见 docs/SPEC.md §3.7），不是从别处抄来的。

    与"使用第三方 API 就必须使用它的 URL"同理：要测试一个结构化输出接口，
    就必须使用该接口的字段名，作者在骨架层面没有表达空间。

    归一化步骤（只影响**比较**，报告仍展示原文）：

    1. 剥离推理标签与 Markdown 代码围栏——它们同样是接口约定的一部分；
    2. 若剩余部分能解析为 JSON 对象，则取其**值的拼接**作为比较对象。

    保留值而非键，是因为**值**才是夹具作者实际写下的内容：
    若连值也被复制，归一化后依然高度相似，检测能力不受影响。
    """
    import json as _json

    text = _REASONING_TAG_RE.sub("", value).strip()
    fenced = _FENCE_RE_FOR_STRINGS.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = _json.loads(text)
    except (ValueError, TypeError):
        return value
    if not isinstance(parsed, dict):
        return value
    return " \u0001 ".join(_flatten_json_values(parsed))


def _flatten_json_values(node: Any) -> list[str]:
    """递归收集 JSON 结构中的所有标量值（按键名排序以保证确定性）。"""
    if isinstance(node, dict):
        collected: list[str] = []
        for key in sorted(node):
            collected.extend(_flatten_json_values(node[key]))
        return collected
    if isinstance(node, list):
        collected = []
        for item in node:
            collected.extend(_flatten_json_values(item))
        return collected
    return [str(node)]


#: 用于在字符串比对前剥离代码围栏。
_FENCE_RE_FOR_STRINGS = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: 用于在字符串比对前剥离推理标签。
_REASONING_TAG_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def best_string_matches(
    target_strings: Sequence[tuple[str, str, int]],
    reference_strings: Sequence[tuple[str, str, int]],
    threshold: float,
    top_n: int,
    min_matched_chars: int,
) -> tuple[list[StringPair], int, int, int]:
    """对每个目标字符串求最相似的上游字符串。

    参数中的三元组为 ``(文件, 内容, 行号)``。返回
    ``(top 列表, 目标串总数, 相似度达阈值的目标串数, 判定为违规的目标串数)``。

    判定分两级：

    * ``ratio >= threshold``：与上游最相似串的相似度达标（用于统计与展示）；
    * **违规**：还需要"实际匹配字符数 ≥ ``min_matched_chars``"。这一条把
      DOI / URL / arXiv 编号 / 作者名 / 纯标识符之类的"事实性短串"（它们天然高度相似，
      却不含任何受保护的表达）排除在失败判据之外，同时不影响长 Prompt 片段的检出。

    先用长度上下界 + ``real_quick_ratio``/``quick_ratio`` 剪枝，避免 O(N*M) 全量比对。
    """
    if not target_strings or not reference_strings:
        return [], len(target_strings), 0, 0

    ordered = sorted(reference_strings, key=lambda item: len(item[1]))
    lengths = [len(item[1]) for item in ordered]
    results: list[StringPair] = []
    threshold_hits = 0
    violations = 0

    for target in target_strings:
        value = target[1]
        size = len(value)
        if size < MIN_STRING_LENGTH:
            continue
        # 整个字符串就是一个 URL → 第三方服务规定的端点，属事实性标识符。
        # 使用 Crossref/OpenAlex 的 API 就必须使用它们的 URL，作者没有表达空间。
        # 只豁免"整个串是 URL"的情形：含 URL 的长文本照常参与比对。
        if URL_LITERAL_RE.match(value.strip()):
            continue
        low = bisect.bisect_left(lengths, max(1, int(size * threshold / (2 - threshold)) - 1))
        high = bisect.bisect_right(lengths, int(size * (2 - threshold) / threshold) + 1)
        candidates = ordered[low:high] if high > low else ordered
        best_ratio = 0.0
        best_ref: tuple[str, str, int] | None = None
        best_matched = 0
        violating = False
        for ref in candidates:
            matcher = difflib.SequenceMatcher(
                None, normalize_for_comparison(value), normalize_for_comparison(ref[1]),
                autojunk=False,
            )
            ceiling = matcher.real_quick_ratio()
            if ceiling <= best_ratio and ceiling < threshold:
                continue
            ceiling = matcher.quick_ratio()
            if ceiling <= best_ratio and ceiling < threshold:
                continue
            ratio = matcher.ratio()
            matched = 0
            if ratio >= threshold:
                matched = sum(block.size for block in matcher.get_matching_blocks())
                if matched >= min_matched_chars:
                    violating = True
            if ratio > best_ratio:
                best_ratio, best_ref, best_matched = ratio, ref, matched
        if best_ref is None:
            continue
        if best_ratio >= threshold:
            threshold_hits += 1
        if violating:
            violations += 1
        results.append(
            StringPair(
                ratio=best_ratio,
                matched=best_matched,
                target=target,
                reference=best_ref,
                violation=violating,
            )
        )

    # **违规项优先展示**：本表按相似度排序后会被截断到 top_n，
    # 而"违规"的判据是相似度 **且** 实际匹配字符数 ≥ 阈值——两者不是同一个排序键。
    # 早先只按相似度排序，出现过"计数说有 3 条违规、表里一条都看不到"的报告缺陷：
    # 一个失败的检查项如果不在报告里指出失败在哪，这个报告就是不可用的。
    results.sort(
        key=lambda item: (
            not item.violation,
            -item.ratio,
            -item.matched,
            item.target[0],
            item.target[2],
        )
    )
    return results[:top_n], len(target_strings), threshold_hits, violations


# --------------------------------------------------------------------------- #
# 检查 1：token n-gram 重叠
# --------------------------------------------------------------------------- #


@dataclass
class NgramOutcome:
    passed: bool
    metric: str
    threshold: str
    details: list[str]


def _longest_common_run_length(
    target_tokens: Sequence[str],
    reference_run_hashes: dict[int, set[int]],
) -> int:
    """返回目标 token 流与参考语料的最长公共连续片段长度的**下界档位**。

    用于把"框架惯用式造成的统计重叠"与"真正的连续搬运"区分开：
    使用同一框架、同一领域的两个项目，8-gram 重叠必然存在（``Field(default=0, ge=0)``、
    ``@field_validator("summary")`` 这类脚手架序列会大量命中），**但随机重叠不会
    形成几十个 token 的连续片段**；连续片段越长，是搬运而非趋同的证据就越强。

    实现上对候选长度做阶梯式探测（而非二分），因为一段共享代码的典型长度
    只需判断量级；阶梯命中即返回，成本低且结果稳定。

    Args:
        target_tokens: 目标文件的 token 序列。
        reference_run_hashes: ``长度 -> 参考语料该长度的 n-gram 哈希集合``。

    Returns:
        命中的最长阶梯长度；无命中返回 0。
    """
    for length in sorted(reference_run_hashes, reverse=True):
        known = reference_run_hashes[length]
        if any(hash(gram) in known for gram in ngrams(target_tokens, length)):
            return length
    return 0


def run_ngram_check(
    target_files: Sequence[TargetFile],
    reference_files: Sequence[RefFile],
    *,
    ngram: int,
    fail_threshold: float,
    file_threshold: float,
    top_n: int,
) -> NgramOutcome:
    """对 ``.py`` 文件做 token n-gram containment / Jaccard 判定。

    判定采用**双重条件**（这是本检查项的核心设计）：

    1. 单文件 containment 超过阈值，**且**
    2. 该文件与参考语料存在长度 ≥ :data:`MIN_COMMON_TOKEN_RUN` 的连续公共 token 片段。

    只满足条件 1 不判失败。理由是实测所得：两个项目若使用同一框架处理同一领域，
    纯脚手架 token 序列（字段声明、装饰器、类型标注）就足以把 containment 推到 5%
    以上——这是**趋同**（scènes à faire），不是抄袭。要求条件 2 作为佐证，
    使该指标从"噪声报警器"变成"有证据的判据"：随机脚手架重叠不会形成
    25 个 token 的连续片段。

    仍保留条件 1 的数值展示（含小文件豁免），以便人工复核。
    """
    reference_ngrams: set[tuple[str, ...]] = set()
    reference_run_hashes: dict[int, set[int]] = {
        length: set() for length in COMMON_RUN_LADDER
    }
    reference_files_used = 0
    for ref in reference_files:
        if ref.text is None or PurePosixPath(ref.rel).suffix not in PY_SUFFIXES:
            continue
        reference_files_used += 1
        ref_tokens = tokenize_python(ref.text)
        reference_ngrams |= ngrams(ref_tokens, ngram)
        for length in COMMON_RUN_LADDER:
            reference_run_hashes[length] |= {hash(gram) for gram in ngrams(ref_tokens, length)}

    per_file: list[tuple[float, str, int, int]] = []
    samples: dict[str, list[tuple[str, ...]]] = {}
    longest_runs: dict[str, int] = {}
    target_ngrams: set[tuple[str, ...]] = set()
    scanned = 0
    for target in target_files:
        if target.text is None or PurePosixPath(target.rel).suffix not in PY_SUFFIXES:
            continue
        scanned += 1
        target_tokens = tokenize_python(target.text)
        file_ngrams = ngrams(target_tokens, ngram)
        target_ngrams |= file_ngrams
        shared = file_ngrams & reference_ngrams
        if shared:
            samples[target.rel] = sorted(shared)[:5]
        longest_runs[target.rel] = _longest_common_run_length(target_tokens, reference_run_hashes)
        per_file.append(
            (containment(file_ngrams, reference_ngrams), target.rel, len(file_ngrams), len(shared))
        )

    overall = containment(target_ngrams, reference_ngrams)
    overall_jaccard = jaccard(target_ngrams, reference_ngrams)
    per_file.sort(key=lambda item: (-item[0], item[1]))
    max_file = per_file[0][0] if per_file else 0.0
    # 小样本文件不参与**containment** 判定：n-gram 总数不足时方差极大，
    # 十几个通用 token 序列即可越过阈值（首轮审计误报了 ports/common.py 的 13.8%）。
    small_files = [item for item in per_file if item[2] < SMALL_FILE_NGRAMS]
    # 判定规则（把"决定性证据"与"统计信号"分开）：
    #
    #   A. 存在 ≥ MIN_COMMON_TOKEN_RUN 的连续公共 token 片段 → **失败**。
    #      连续几十个 token 与上游逐字一致，无法用"框架趋同"解释，
    #      且该判据不受文件大小与 containment 阈值影响——
    #      否则把复制内容塞进小文件即可同时躲过阈值与小文件豁免。
    #   B. containment 超阈值但无长连续片段 → 通过，但在报告中列为"待人工复核"。
    #      这是同框架 + 同领域下的趋同（scènes à faire），不是复制。
    failing_files = [
        item for item in per_file if longest_runs.get(item[1], 0) >= MIN_COMMON_TOKEN_RUN
    ]
    suspect_files = [
        item
        for item in per_file
        if item[0] > file_threshold
        and item[2] >= SMALL_FILE_NGRAMS
        and longest_runs.get(item[1], 0) < MIN_COMMON_TOKEN_RUN
    ]
    passed = overall <= fail_threshold and not failing_files

    max_run = max(longest_runs.values(), default=0)
    metric = (
        f"整体 containment {overall:.4%}（Jaccard {overall_jaccard:.4%}）；"
        f"单文件最高 {max_file:.4%}；最长公共 token 连续片段 {max_run}"
    )
    threshold = (
        f"整体 containment ≤ {fail_threshold:.2%}；且**任一文件**不得存在 "
        f"≥ {MIN_COMMON_TOKEN_RUN} token 的连续公共片段（该判据不受文件大小豁免）；"
        f"单文件 containment > {file_threshold:.2%} 且无长连续片段者列为待复核"
    )

    details: list[str] = []
    details.append(
        f"- 上游参考 n-gram 池：**{len(reference_ngrams):,}** 个（来自 {reference_files_used} 个 `.py` 文件）"
    )
    details.append(f"- 目标 n-gram 池：**{len(target_ngrams):,}** 个（扫描 {scanned} 个 `.py` 文件）")
    details.append(f"- 交集：**{len(target_ngrams & reference_ngrams):,}** 个")
    details.append("")
    details.append(f"#### 重叠最高的前 {top_n} 个目标文件")
    details.append("")
    if per_file:
        rows = [
            (
                str(index),
                _md_code(rel),
                f"{ratio:.4%}",
                f"{shared:,}",
                f"{total:,}",
                str(longest_runs.get(rel, 0)),
            )
            for index, (ratio, rel, total, shared) in enumerate(per_file[:top_n], start=1)
        ]
        details.extend(
            _md_table(
                ["#", "目标文件", "containment", "共有 n-gram", "文件 n-gram 总数", "最长公共连续片段"],
                rows,
            )
        )
    else:
        details.append("_目标项目中未发现可比对的源码文件。_")
    if failing_files:
        details.append("")
        details.append(
            f"**❗ 判定失败的文件（containment > {file_threshold:.2%} 且最长公共片段 ≥ "
            f"{MIN_COMMON_TOKEN_RUN} token）：**"
        )
        for ratio, rel, _total, _shared in failing_files[:top_n]:
            details.append(
                f"- {_md_code(rel)} → containment {ratio:.4%}，最长公共片段 "
                f"{longest_runs.get(rel, 0)} token"
            )
    if suspect_files:
        details.append("")
        details.append(
            f"**✅ 统计超标但判定通过的文件（containment > {file_threshold:.2%}，"
            f"但最长公共片段 < {MIN_COMMON_TOKEN_RUN} token）：**"
        )
        for ratio, rel, _total, _shared in suspect_files[:top_n]:
            details.append(
                f"- {_md_code(rel)} → containment {ratio:.4%}，最长公共片段 "
                f"{longest_runs.get(rel, 0)} token"
            )
        details.append("")
        details.append(
            "> 这类文件的重叠来自**框架与领域的趋同**（同用 pydantic 表达同一领域概念时，"
            "字段声明、装饰器与类型标注等脚手架序列必然重合），属于著作权法上的"
            "「表达与思想合并」情形，不构成复制证据。上表的共享 n-gram 示例可人工复核。"
        )
    if samples:
        details.append("")
        details.append("#### 共享 n-gram 示例（供人工判断是否为通用写法）")
        details.append("")
        rows = []
        for _ratio, rel, _total, _shared in per_file[:5]:
            for gram in samples.get(rel, [])[:3]:
                rows.append((_md_code(rel), _md_code(" ".join(gram))))
        if rows:
            details.extend(_md_table(["目标文件", "共享 n-gram（token 序列）"], rows))
    if small_files:
        details.append("")
        details.append(
            f"> ℹ️ 另有 {len(small_files)} 个文件的 n-gram 总数不足 {SMALL_FILE_NGRAMS}，"
            "**不参与单文件判定**：小样本下 containment 方差极大（十几个通用 token 序列"
            "即可超过 5%），属于统计假象而非复制证据。这些文件仍列在上表中，"
            "其最高值为 "
            + f"{small_files[0][0]:.4%}（{_md_code(small_files[0][1])}）"
            + "，请结合共享 n-gram 示例人工核验。"
        )

    return NgramOutcome(
        passed=passed, metric=metric, threshold=threshold, details=details
    )


# --------------------------------------------------------------------------- #
# 检查 2：逐行精确复制
# --------------------------------------------------------------------------- #


@dataclass
class LineMatch:
    """一条逐字相同的代码行。"""

    target_rel: str
    target_line: int
    text: str
    tier: str
    occurrences: list[tuple[str, int]]


@dataclass
class LineOutcome:
    passed: bool
    metric: str
    threshold: str
    details: list[str]


def collect_common_identifiers(reference_files: Sequence[RefFile]) -> frozenset[str]:
    """统计上游语料中"通用标识符"的集合（文档频率 ≥ :data:`COMMON_IDENTIFIER_DF`）。

    同时无条件收录 Python 关键字与内置名——它们在任何语料里都是通用的，
    而小语料上算出的文档频率并不可靠。
    """
    document_frequency: dict[str, int] = {}
    files = 0
    for ref in reference_files:
        if ref.text is None or PurePosixPath(ref.rel).suffix not in PY_SUFFIXES:
            continue
        files += 1
        for name in set(IDENTIFIER_RE.findall(ref.text)):
            document_frequency[name] = document_frequency.get(name, 0) + 1

    common = set(keyword.kwlist) | set(keyword.softkwlist) | set(dir(builtins))
    if files:
        threshold = max(1, math.ceil(files * COMMON_IDENTIFIER_DF))
        common |= {name for name, count in document_frequency.items() if count >= threshold}
    return frozenset(common)


def has_distinctive_content(stripped: str, common_identifiers: frozenset[str]) -> bool:
    """这一行是否含有**承载作者选择**的内容。

    两类才算：

    1. 至少一个**非通用**标识符（命名选择）；
    2. 至少一个字面量字符串达到 :data:`MIN_STRING_LENGTH`（措辞选择）。

    只有形如 ``assert len(result) == 1``、``self.settings = settings`` 这样
    完全由通用词汇与语法构成的行才被判为无区分性。这条判据把检查对准了
    真正的信号——**复制会带来上游的特征标识符**，而趋同只会带来通用词。
    """
    # 先把字面量字符串整体替换掉再取标识符：字符串**内部**的文本是字面量内容，
    # 不是标识符。不剥离会重复计数，而且会绕过字符串检查项更细致的判据
    # （相似度比例 + 实际匹配字符数），把"CSV 列名"这类第三方数据格式的字段名
    # 误判成命名选择。
    without_literals = STRING_LITERAL_RE.sub('""', stripped)
    for name in IDENTIFIER_RE.findall(without_literals):
        if len(name) > 2 and name not in common_identifiers:
            return True
    # 长字面量由字符串检查项负责判定，这里只做"存在即视为有内容"的粗筛
    return any(
        len(match.group(0)) - 2 >= MIN_STRING_LENGTH
        for match in STRING_LITERAL_RE.finditer(stripped)
    )


def classify_line(stripped: str, common_identifiers: frozenset[str] = frozenset()) -> str:
    """把一行代码归入 样板 / 通用惯用式 / 实质性 三档。

    Args:
        stripped: 去掉首尾空白后的行内容。
        common_identifiers: 上游语料中的通用标识符集合。传入空集合时退化为
            只按 :data:`BOILERPLATE_LINE_RE` 与 :data:`GENERIC_IDIOM_RES` 判断。
    """
    if BOILERPLATE_LINE_RE.match(stripped):
        return TIER_BOILERPLATE
    if any(pattern.match(stripped) for pattern in GENERIC_IDIOM_RES):
        return TIER_GENERIC
    # 没有任何"作者做过选择"的痕迹 → 这一行不可能是表达层面的复制证据
    if common_identifiers and not has_distinctive_content(stripped, common_identifiers):
        return TIER_GENERIC
    return TIER_SUBSTANTIVE


def _aligned_blocks(
    matches: Sequence[LineMatch], max_occurrences: int = 8
) -> list[tuple[int, str, str, int, int]]:
    """找出目标与上游"行号同步递增"的对齐片段（复制的强证据）。

    若目标的第 t 行 = 上游的 r 行、且 t+1 行 = r+1 行……则这一组命中的
    ``(ref_line - target_line)`` 偏移量恒定。返回 ``(长度, 目标文件, 上游文件, 目标起始行, 上游起始行)``。
    """
    groups: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for match in matches:
        for ref_rel, ref_line in match.occurrences[:max_occurrences]:
            key = (match.target_rel, ref_rel, ref_line - match.target_line)
            groups.setdefault(key, []).append((match.target_line, ref_line))

    blocks: list[tuple[int, str, str, int, int]] = []
    for (target_rel, ref_rel, _delta), pairs in groups.items():
        pairs.sort()
        run_start = 0
        for index in range(1, len(pairs) + 1):
            continues = (
                index < len(pairs)
                and pairs[index][0] == pairs[index - 1][0] + 1
                and pairs[index][1] == pairs[index - 1][1] + 1
            )
            if continues:
                continue
            length = index - run_start
            if length >= 2:
                blocks.append((length, target_rel, ref_rel, pairs[run_start][0], pairs[run_start][1]))
            run_start = index
    blocks.sort(key=lambda item: (-item[0], item[1], item[3]))
    return blocks


def run_line_check(
    target_files: Sequence[TargetFile],
    reference_files: Sequence[RefFile],
    *,
    max_copied_lines: int,
    min_copied_block: int,
    top_n: int,
) -> LineOutcome:
    """去空白后的非空行精确匹配（忽略 <12 字符的行、通用样板与通用惯用式）。

    失败条件（二者之一）：
    * 实质性逐字相同行数 > ``max_copied_lines``；
    * 存在长度 ≥ ``min_copied_block`` 的"行号同步递增"对齐片段（整块搬运的强证据）。
    """

    common_identifiers = collect_common_identifiers(reference_files)
    index: dict[str, list[tuple[str, int]]] = {}
    for ref in reference_files:
        if ref.text is None or PurePosixPath(ref.rel).suffix not in PY_SUFFIXES:
            continue
        for lineno, raw in enumerate(ref.text.splitlines(), start=1):
            stripped = raw.strip()
            if len(stripped) < MIN_LINE_LENGTH or BOILERPLATE_LINE_RE.match(stripped):
                continue
            index.setdefault(stripped, []).append((ref.rel, lineno))

    matches: list[LineMatch] = []
    boilerplate_hits = 0
    scanned_lines = 0
    for target in target_files:
        if target.text is None or PurePosixPath(target.rel).suffix not in PY_SUFFIXES:
            continue
        for lineno, raw in enumerate(target.text.splitlines(), start=1):
            stripped = raw.strip()
            if len(stripped) < MIN_LINE_LENGTH:
                continue
            scanned_lines += 1
            if BOILERPLATE_LINE_RE.match(stripped):
                if stripped in index:
                    boilerplate_hits += 1
                continue
            occurrences = index.get(stripped)
            if occurrences:
                matches.append(
                    LineMatch(
                        target_rel=target.rel,
                        target_line=lineno,
                        text=stripped,
                        tier=classify_line(stripped, common_identifiers),
                        occurrences=list(occurrences),
                    )
                )

    substantive = [match for match in matches if match.tier == TIER_SUBSTANTIVE]
    generic = [match for match in matches if match.tier == TIER_GENERIC]
    blocks = _aligned_blocks(matches)
    longest_block = blocks[0][0] if blocks else 0
    failing_blocks = [block for block in blocks if block[0] >= min_copied_block]

    passed = len(substantive) <= max_copied_lines and not failing_blocks
    metric = (
        f"逐字相同行 {len(matches)}（实质性 {len(substantive)} / 通用惯用式 {len(generic)}）；"
        f"最长对齐片段 {longest_block} 行"
    )
    threshold = f"实质性 ≤ {max_copied_lines} 行 且 对齐片段 < {min_copied_block} 行"

    details: list[str] = []
    details.append(
        f"- 上游可比对代码行索引：**{len(index):,}** 条唯一非空行（`.py`，已剔除 <{MIN_LINE_LENGTH} 字符与通用样板）"
    )
    details.append(f"- 目标侧扫描的非空行：**{scanned_lines:,}** 行")
    details.append(f"- 命中（逐字相同）：**{len(matches)}** 行，其中 **实质性 {len(substantive)}** 行、通用惯用式 {len(generic)} 行")
    details.append(f"- 属于通用样板（import / `if __name__` 等）而完全忽略：**{boilerplate_hits}** 行")
    details.append(f"- 行号同步递增的对齐片段：**{len(blocks)}** 段，最长 **{longest_block}** 行")
    details.append("")
    details.append(
        "> 说明：`通用惯用式`（类型守卫、异常捕获、单关键字语句、关键字实参行等）"
        "与 <12 字符规则同性质，属于“必然重复”，只展示不判失败；"
        "真正的复制证据是 **实质性行** 与 **对齐片段**。"
    )
    details.append("")
    details.append(f"#### 前 {top_n} 条匹配")
    details.append("")
    if matches:
        rows = []
        for position, match in enumerate(
            sorted(matches, key=lambda item: (item.tier != TIER_SUBSTANTIVE, item.target_rel, item.target_line))[
                :top_n
            ],
            start=1,
        ):
            ref_rel, ref_line = match.occurrences[0]
            rows.append(
                (
                    str(position),
                    _md_code(match.tier),
                    f"{_md_code(match.target_rel)}:{match.target_line}",
                    f"{_md_code(ref_rel)}:{ref_line}",
                    _md_code(match.text),
                )
            )
        details.extend(_md_table(["#", "层级", "目标位置", "上游位置（首个）", "行内容预览"], rows))
    else:
        details.append("_未发现逐字复制的代码行。_")
    if blocks:
        details.append("")
        details.append("#### 行号同步递增的对齐片段")
        details.append("")
        rows = [
            (
                str(position),
                str(length),
                _md_code(target_rel),
                f"{target_line}–{target_line + length - 1}",
                _md_code(ref_rel),
                f"{ref_line}–{ref_line + length - 1}",
            )
            for position, (length, target_rel, ref_rel, target_line, ref_line) in enumerate(blocks[:top_n], start=1)
        ]
        details.extend(
            _md_table(["#", "长度", "目标文件", "目标行区间", "上游文件", "上游行区间"], rows)
        )
        if failing_blocks:
            details.append("")
            details.append(f"**超过对齐片段阈值 {min_copied_block} 行的片段即为失败依据。**")

    return LineOutcome(passed=passed, metric=metric, threshold=threshold, details=details)


# --------------------------------------------------------------------------- #
# 检查 3：字符串字面量相似度 + 专有标识符
# --------------------------------------------------------------------------- #


@dataclass
class StringOutcome:
    passed: bool
    metric: str
    threshold: str
    details: list[str]


def _collect_python_strings(
    files: Sequence[TargetFile] | Sequence[RefFile],
) -> tuple[list[tuple[str, str, int]], int]:
    """收集 ``.py`` 中长度达标的非 docstring 字符串字面量。"""
    collected: list[tuple[str, str, int]] = []
    fallback_files = 0
    for item in files:
        if item.text is None or PurePosixPath(item.rel).suffix not in PY_SUFFIXES:
            continue
        literals, used_ast = extract_string_literals(item.text)
        if not used_ast:
            fallback_files += 1
        for value, lineno in literals:
            if len(value) >= MIN_STRING_LENGTH:
                collected.append((item.rel, value, lineno))
    return collected, fallback_files


@dataclass(frozen=True)
class ProseSpan:
    """文本区间及其性质：注释 / docstring / 普通字符串字面量。"""

    start: int
    end: int
    kind: str  # "comment" | "docstring" | "string"


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _fallback_prose_spans(text: str) -> list[ProseSpan]:
    """不依赖解析器的注释区间扫描（引号内的 ``#`` 不算注释）。"""
    spans: list[ProseSpan] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\n":
            quote, escaped = None, False
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
        elif char == "#":
            end = text.find("\n", index)
            end = len(text) if end == -1 else end
            spans.append(ProseSpan(index, end, "comment"))
            index = end
            continue
        index += 1
    return spans


def _docstring_spans(text: str) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    starts = _line_starts(text)

    def absolute(line: int, column: int) -> int:
        if 1 <= line <= len(starts):
            return starts[line - 1] + column
        return -1

    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            value = first.value
            spans.append(
                (
                    absolute(value.lineno, value.col_offset),
                    absolute(getattr(value, "end_lineno", value.lineno), getattr(value, "end_col_offset", 0)),
                )
            )
    return spans


def python_prose_spans(text: str) -> list[ProseSpan]:
    """收集 ``.py`` 中的注释 / docstring / 字符串字面量区间（tokenize 优先）。"""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return _fallback_prose_spans(text)
    starts = _line_starts(text)

    def absolute(position: tuple[int, int]) -> int:
        row, column = position
        if 1 <= row <= len(starts):
            return starts[row - 1] + column
        return -1

    docstrings = _docstring_spans(text)
    spans: list[ProseSpan] = []
    for token in tokens:
        if token.type == tokenize.COMMENT:
            spans.append(ProseSpan(absolute(token.start), absolute(token.end), "comment"))
        elif token.type == tokenize.STRING:
            start, end = absolute(token.start), absolute(token.end)
            kind = "docstring" if any(low <= start and end <= high for low, high in docstrings) else "string"
            spans.append(ProseSpan(start, end, kind))
    return spans


def classify_offset(spans: Sequence[ProseSpan], offset: int) -> str:
    """判断某个偏移量落在注释 / docstring / 字符串 / 普通代码中。"""
    for span in spans:
        if span.start <= offset < span.end:
            return span.kind
    return "code"


def run_string_check(
    target_files: Sequence[TargetFile],
    reference_files: Sequence[RefFile],
    *,
    string_threshold: float,
    min_matched_chars: int,
    top_n: int,
    scan_identifiers: bool = True,
    strict_identifiers: bool = False,
) -> StringOutcome:
    """Prompt 泄露检测：长字符串字面量的最相似匹配 + 上游专有标识符扫描。

    专有标识符采用分级判定：出现在**注释 / docstring** 中的提及（例如许可归属声明、
    "本实现不依赖 XXX" 的说明）记为 ⚠️ 警告；出现在**代码或普通字符串**中的记为一处违规。
    ``strict_identifiers=True`` 时两者都判失败。
    """
    reference_strings, ref_fallback = _collect_python_strings(reference_files)
    target_strings, tgt_fallback = _collect_python_strings(target_files)

    top_matches, _target_total, hit_count, violations = best_string_matches(
        target_strings, reference_strings, string_threshold, top_n, min_matched_chars
    )

    failures: list[tuple[str, int, str, str, str]] = []
    warnings: list[tuple[str, int, str, str, str]] = []
    if scan_identifiers:
        needles = [(token, token.lower()) for token in PROHIBITED_IDENTIFIERS]
        for item in target_files:
            suffix = PurePosixPath(item.rel).suffix
            if item.text is None or suffix not in CODE_SCAN_SUFFIXES:
                continue
            if is_audit_tooling(item.rel, item.text):
                continue
            spans = python_prose_spans(item.text) if suffix in PY_SUFFIXES else _fallback_prose_spans(item.text)
            starts = _line_starts(item.text)
            text_lower = item.text.lower()
            for token, needle in needles:
                search_from = 0
                while True:
                    position = text_lower.find(needle, search_from)
                    if position < 0:
                        break
                    search_from = position + 1
                    line = bisect.bisect_right(starts, position)
                    kind = classify_offset(spans, position)
                    record = (item.rel, line, token, kind, item.text.splitlines()[line - 1].strip())
                    if kind in {"comment", "docstring"} and not strict_identifiers:
                        warnings.append(record)
                    else:
                        failures.append(record)

    passed = violations == 0 and not failures
    metric = (
        f"违规相似串 {violations} 条（≥{string_threshold:.2f} 相似串 {hit_count} 条）；"
        f"专有标识符违规 {len(failures)} 处（说明性提及 {len(warnings)} 处）"
    )
    threshold = (
        f"相似度 ≥ {string_threshold:.2f} 且匹配字符 ≥ {min_matched_chars} 的相似串 = 0 且 代码中的标识符违规 = 0"
    )

    details: list[str] = []
    details.append(f"- 上游字符串字面量（≥{MIN_STRING_LENGTH} 字符，已跳过 docstring）：**{len(reference_strings):,}** 条")
    details.append(f"- 目标字符串字面量（≥{MIN_STRING_LENGTH} 字符，已跳过 docstring）：**{len(target_strings):,}** 条")
    details.append(f"- 与上游最相似串相似度 ≥ {string_threshold:.2f} 的目标串：**{hit_count}** 条")
    details.append(
        f"- 其中**判定为违规**（实际匹配字符数 ≥ {min_matched_chars}）：**{violations}** 条"
    )
    details.append(
        "- 说明：相似度高但匹配字符数不足的字符串，多为 DOI / URL / arXiv 编号 / 作者名 / 纯标识符"
        "等**事实性短串**（DOI 天生彼此相似），不含受保护表达，因此不判失败；"
        f"可用 `--min-matched-chars` 调整该门槛。"
    )
    details.append(
        f"- 专有标识符：违规 **{len(failures)}** 处、说明性提及（注释 / docstring）**{len(warnings)}** 处"
        + ("（`--strict-identifiers` 已开启，二者均判失败）" if strict_identifiers else "")
    )
    if ref_fallback or tgt_fallback:
        details.append(
            f"- ⚠️ ast 解析失败而回退正则提取的文件：上游 {ref_fallback} 个 / 目标 {tgt_fallback} 个"
            "（回退模式下无法识别 docstring，可能与检查 1 重复计数）"
        )
    details.append("")
    details.append(f"#### 相似度最高的前 {top_n} 对字符串（目标 ← 上游）")
    details.append("")
    if top_matches:
        rows = []
        for position, pair in enumerate(top_matches, start=1):
            flag = "❗" if pair.violation else ("·" if pair.ratio >= string_threshold else "")
            rows.append(
                (
                    str(position),
                    f"{pair.ratio:.3f} {flag}".strip(),
                    str(pair.matched),
                    f"{_md_code(pair.target[0])}:{pair.target[2]}",
                    f"{_md_code(pair.reference[0])}:{pair.reference[2]}",
                    _md_code(pair.target[1]),
                    _md_code(pair.reference[1]),
                )
            )
        details.extend(
            _md_table(
                ["#", "相似度", "匹配字符", "目标位置", "上游位置", "目标字符串", "上游字符串"],
                rows,
            )
        )
        details.append("")
        details.append("_标记说明：❗ = 违规（相似度与匹配字符数同时达标）；· = 相似度达标但匹配字符数不足，不计为违规。_")
    else:
        details.append("_没有任何可比对的字符串对。_")
    details.append("")
    details.append("#### 上游专有标识符专项检查")
    details.append("")
    details.append(
        "扫描目标 `.py` / 配置文件中是否出现 "
        + "、".join(_md_code(token) for token in PROHIBITED_IDENTIFIERS)
        + "（大小写不敏感子串匹配；审计工具自身文件已排除）。"
        "出现在注释或 docstring 中的**归属声明 / 说明性提及**记为警告，出现在代码或普通字符串中的记为违规。"
    )
    details.append("")
    details.append("**违规明细（代码 / 普通字符串）**")
    details.append("")
    if failures:
        rows = [
            (str(position), _md_code(rel), str(line), _md_code(token), _md_code(kind), _md_code(content))
            for position, (rel, line, token, kind, content) in enumerate(failures[:top_n], start=1)
        ]
        details.extend(_md_table(["#", "文件", "行号", "命中", "位置性质", "行内容"], rows))
    else:
        details.append("_无。_")
    details.append("")
    details.append("**说明性提及（注释 / docstring，不判失败）**")
    details.append("")
    if warnings:
        rows = [
            (str(position), _md_code(rel), str(line), _md_code(token), _md_code(kind), _md_code(content))
            for position, (rel, line, token, kind, content) in enumerate(warnings[:top_n], start=1)
        ]
        details.extend(_md_table(["#", "文件", "行号", "命中", "位置性质", "行内容"], rows))
        details.append("")
        details.append(
            "_上述提及属于归属声明或说明性文字（例如指明设计范式来源、声明不依赖上游包），"
            "不含受保护的表达，因此不计为违规；如需最严格口径可加 `--strict-identifiers`。_"
        )
    else:
        details.append("_无。_")

    return StringOutcome(passed=passed, metric=metric, threshold=threshold, details=details)


# --------------------------------------------------------------------------- #
# 检查 4：依赖审计
# --------------------------------------------------------------------------- #


@dataclass
class DependencyOutcome:
    passed: bool
    metric: str
    threshold: str
    details: list[str]


def _dependencies_from_pyproject(path: Path, data: bytes) -> tuple[list[tuple[str, str]], str | None]:
    """返回 ``[(requirement, 来源标签)]``；解析失败时返回 ``([], 错误信息)``。"""
    try:
        document = tomllib.loads(data.decode("utf-8", "replace"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return [], f"{path.name}: TOML 解析失败（{exc}）"

    results: list[tuple[str, str]] = []
    project = document.get("project", {})
    if isinstance(project, dict):
        for requirement in project.get("dependencies", []) or []:
            results.append((str(requirement), "[project].dependencies"))
        optional = project.get("optional-dependencies", {}) or {}
        if isinstance(optional, dict):
            for group, requirements in optional.items():
                for requirement in requirements or []:
                    results.append((str(requirement), f"[project.optional-dependencies.{group}]"))
    groups = document.get("dependency-groups", {}) or {}
    if isinstance(groups, dict):
        for group, requirements in groups.items():
            for requirement in requirements or []:
                if isinstance(requirement, str):
                    results.append((requirement, f"[dependency-groups.{group}]"))
    build = document.get("build-system", {}) or {}
    if isinstance(build, dict):
        for requirement in build.get("requires", []) or []:
            results.append((str(requirement), "[build-system].requires"))
    return results, None


def _dependencies_from_requirements(data: bytes) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for raw in data.decode("utf-8", "replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = DEP_NAME_RE.match(line)
        if match:
            results.append((line, "requirements"))
    return results


def _imports_from_python(text: str) -> list[str]:
    """返回源码中出现的顶层模块名（ast 优先，失败回退正则）。"""
    modules: list[str] = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", text, re.MULTILINE):
            modules.append(match.group(1))
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
    # 动态导入：importlib.import_module("x") / __import__("x")
    for match in re.finditer(r"(?:import_module|__import__)\(\s*[\"']([\w.]+)[\"']", text):
        modules.append(match.group(1))
    return modules


def run_dependency_check(target_files: Sequence[TargetFile]) -> DependencyOutcome:
    """检查 pyproject.toml / requirements*.txt / import 语句中的禁用依赖。"""
    violations: list[tuple[str, str, str, str]] = []
    warnings: list[tuple[str, str, str]] = []
    scanned_sources: list[str] = []

    for item in target_files:
        name = PurePosixPath(item.rel).name
        suffix = PurePosixPath(item.rel).suffix
        is_pyproject = name == "pyproject.toml"
        is_requirements = name.startswith("requirements") and suffix == ".txt"
        if not (is_pyproject or is_requirements):
            continue
        if is_audit_tooling(item.rel, item.text):
            continue
        if is_pyproject:
            entries, error = _dependencies_from_pyproject(item.abspath, item.data)
            if error:
                warnings.append((item.rel, error, ""))
            scanned_sources.append(item.rel)
        else:
            entries = _dependencies_from_requirements(item.data)
            scanned_sources.append(item.rel)
        for requirement, origin in entries:
            match = DEP_NAME_RE.match(requirement.strip())
            if not match:
                continue
            module = match.group(0)
            banned = _dep_hits(module, BANNED_DEPENDENCIES)
            if banned:
                violations.append((item.rel, origin, requirement, banned))
                continue
            warned = _dep_hits(module, WARN_DEPENDENCIES)
            if warned:
                warnings.append((item.rel, requirement, warned))

    for item in target_files:
        if item.text is None or PurePosixPath(item.rel).suffix not in PY_SUFFIXES:
            continue
        if is_audit_tooling(item.rel, item.text):
            continue
        for module in _imports_from_python(item.text):
            root = module.split(".")[0]
            banned = _dep_hits(root, BANNED_DEPENDENCIES)
            if banned:
                violations.append((item.rel, "import", module, banned))
                continue
            warned = _dep_hits(root, WARN_DEPENDENCIES)
            if warned:
                warnings.append((item.rel, f"import {module}", warned))

    passed = not violations
    metric = f"禁用依赖 {len(violations)} 处；警告 {len(warnings)} 处"
    threshold = "禁用依赖 = 0"

    details: list[str] = []
    details.append("- 禁用清单（命中即失败）：" + "、".join(_md_code(name) for name in BANNED_DEPENDENCIES))
    details.append("- 仅警告：" + "、".join(_md_code(name) for name in WARN_DEPENDENCIES))
    details.append(f"- 已扫描的清单文件：{', '.join(_md_code(name) for name in scanned_sources) or '（无）'}")
    details.append("- 审计工具自身文件（文件名含 `similarity_audit`、`tools/README.md`）已排除，避免自我指控。")
    details.append("")
    details.append("#### 违规明细")
    details.append("")
    if violations:
        rows = [
            (str(index), _md_code(rel), _md_code(origin), _md_code(requirement), _md_code(reason))
            for index, (rel, origin, requirement, reason) in enumerate(violations[:DEFAULT_TOP_N], start=1)
        ]
        details.extend(_md_table(["#", "文件", "位置", "依赖/模块", "说明"], rows))
    else:
        details.append("_未发现禁用依赖。_")
    details.append("")
    details.append("#### 警告明细（不判失败）")
    details.append("")
    if warnings:
        rows = [
            (str(index), _md_code(rel), _md_code(requirement), _md_code(reason))
            for index, (rel, requirement, reason) in enumerate(warnings[:DEFAULT_TOP_N], start=1)
        ]
        details.extend(_md_table(["#", "文件", "依赖/模块", "说明"], rows))
    else:
        details.append("_无警告。_")

    return DependencyOutcome(passed=passed, metric=metric, threshold=threshold, details=details)


# --------------------------------------------------------------------------- #
# 检查 5：资产 / 文档审计
# --------------------------------------------------------------------------- #


@dataclass
class AssetOutcome:
    passed: bool
    metric: str
    threshold: str
    details: list[str]


def run_asset_check(
    target_files: Sequence[TargetFile],
    reference_files: Sequence[RefFile],
    *,
    doc_ngram: int,
    fail_threshold: float,
    file_threshold: float,
    top_n: int,
) -> AssetOutcome:
    """SHA-256 内容哈希比对 + `.md` 文档的 token n-gram 检查。"""
    reference_hashes: dict[str, list[str]] = {}
    for ref in reference_files:
        if not ref.data:  # 空文件哈希相同无信息量，避免假阳性
            continue
        reference_hashes.setdefault(ref.sha256, []).append(ref.rel)

    hash_matches: list[tuple[str, str, int]] = []
    for item in target_files:
        if not item.data:
            continue
        digest = hashlib.sha256(item.data).hexdigest()
        for ref_rel in reference_hashes.get(digest, []):
            hash_matches.append((item.rel, ref_rel, len(item.data)))

    reference_ngrams: set[tuple[str, ...]] = set()
    reference_docs = 0
    for ref in reference_files:
        if ref.text is None:
            continue
        reference_docs += 1
        reference_ngrams |= ngrams(tokenize_text(ref.text), doc_ngram)

    doc_rows: list[tuple[float, str, int, int]] = []
    target_doc_ngrams: set[tuple[str, ...]] = set()
    for item in target_files:
        if item.text is None or PurePosixPath(item.rel).suffix not in DOC_SUFFIXES:
            continue
        file_ngrams = ngrams(tokenize_text(item.text), doc_ngram)
        target_doc_ngrams |= file_ngrams
        doc_rows.append(
            (containment(file_ngrams, reference_ngrams), item.rel, len(file_ngrams), len(file_ngrams & reference_ngrams))
        )

    doc_rows.sort(key=lambda row: (-row[0], row[1]))
    doc_overall = containment(target_doc_ngrams, reference_ngrams)
    doc_jaccard = jaccard(target_doc_ngrams, reference_ngrams)
    doc_max = doc_rows[0][0] if doc_rows else 0.0
    doc_failing = [row for row in doc_rows if row[0] > file_threshold]

    passed = not hash_matches and doc_overall <= fail_threshold and not doc_failing
    metric = f"哈希相同资产 {len(hash_matches)} 个；文档整体 containment {doc_overall:.4%}（最高 {doc_max:.4%}）"
    threshold = f"哈希相同 = 0 且 文档整体 ≤ {fail_threshold:.2%} 且 单文档 ≤ {file_threshold:.2%}"

    details: list[str] = []
    details.append("#### 5.1 资产内容哈希（SHA-256）")
    details.append("")
    details.append(f"- 上游可比对 blob：**{sum(len(v) for v in reference_hashes.values()):,}** 个（已跳过 0 字节文件）")
    details.append(f"- 目标文件哈希相同者：**{len(hash_matches)}** 个")
    details.append("")
    if hash_matches:
        rows = [
            (str(index), _md_code(target_rel), _md_code(ref_rel), f"{size:,} B")
            for index, (target_rel, ref_rel, size) in enumerate(hash_matches[:top_n], start=1)
        ]
        details.extend(_md_table(["#", "目标文件", "上游文件", "大小"], rows))
    else:
        details.append("_未发现与上游 git 跟踪文件内容完全相同的资产。_")
    details.append("")
    details.append(f"#### 5.2 文档 token n-gram（n={doc_ngram}）")
    details.append("")
    details.append(f"- 上游文本文件参与比对：**{reference_docs:,}** 个，n-gram 池 **{len(reference_ngrams):,}** 个")
    details.append(f"- 目标 `.md`/`.rst` 文件：**{len(doc_rows)}** 个，n-gram 池 **{len(target_doc_ngrams):,}** 个")
    details.append(f"- 整体 containment **{doc_overall:.4%}**，Jaccard **{doc_jaccard:.4%}**")
    details.append("")
    if doc_rows:
        rows = [
            (str(index), _md_code(rel), f"{ratio:.4%}", f"{shared:,}", f"{total:,}")
            for index, (ratio, rel, total, shared) in enumerate(doc_rows[:top_n], start=1)
        ]
        details.extend(_md_table(["#", "目标文档", "containment", "共有 n-gram", "文档 n-gram 总数"], rows))
    else:
        details.append("_目标项目中未发现 `.md`/`.rst` 文档。_")

    return AssetOutcome(passed=passed, metric=metric, threshold=threshold, details=details)


# --------------------------------------------------------------------------- #
# 审计主流程
# --------------------------------------------------------------------------- #


def run_audit(
    target: Path,
    reference: Path,
    *,
    report_path: Path | None,
    ngram: int = DEFAULT_NGRAM,
    doc_ngram: int = DEFAULT_DOC_NGRAM,
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    file_threshold: float = DEFAULT_FILE_THRESHOLD,
    string_threshold: float = DEFAULT_STRING_THRESHOLD,
    min_matched_chars: int = DEFAULT_MIN_MATCHED_CHARS,
    max_copied_lines: int = DEFAULT_MAX_COPIED_LINES,
    min_copied_block: int = DEFAULT_MIN_COPIED_BLOCK,
    top_n: int = DEFAULT_TOP_N,
    excludes: Sequence[str] = (),
    strict_identifiers: bool = False,
) -> AuditResult:
    """执行全部检查，返回结构化结果（不写文件）。"""
    if not target.is_dir():
        raise AuditError(f"目标路径不存在或不是目录：{target}")
    if not reference.is_dir():
        raise AuditError(f"参考路径不存在或不是目录：{reference}")
    if target.resolve() == reference.resolve():
        raise AuditError("目标路径与参考路径相同，无法审计")

    if not (reference / ".git").exists():
        raise AuditError(f"参考路径不是 git 仓库（缺少 .git）：{reference}")

    head_commit = git_head_commit(reference)
    head_subject = git_head_subject(reference)
    reference_files, skipped, source = load_reference_files(reference)
    if not reference_files:
        raise AuditError(f"上游 HEAD（{head_commit[:12]}）中没有可读取的跟踪文件：{reference}")

    target_files = iter_target_files(target, report_path=report_path, excludes=excludes)

    ngram_outcome = run_ngram_check(
        target_files,
        reference_files,
        ngram=ngram,
        fail_threshold=fail_threshold,
        file_threshold=file_threshold,
        top_n=top_n,
    )
    line_outcome = run_line_check(
        target_files,
        reference_files,
        max_copied_lines=max_copied_lines,
        min_copied_block=min_copied_block,
        top_n=top_n,
    )
    string_outcome = run_string_check(
        target_files,
        reference_files,
        string_threshold=string_threshold,
        min_matched_chars=min_matched_chars,
        top_n=top_n,
        strict_identifiers=strict_identifiers,
    )
    dependency_outcome = run_dependency_check(target_files)
    asset_outcome = run_asset_check(
        target_files,
        reference_files,
        doc_ngram=doc_ngram,
        fail_threshold=fail_threshold,
        file_threshold=file_threshold,
        top_n=top_n,
    )

    checks = [
        CheckResult(
            key="ngram",
            title=f"1. Token n-gram 重叠（n={ngram}）",
            passed=ngram_outcome.passed,
            metric=ngram_outcome.metric,
            threshold=ngram_outcome.threshold,
            details=ngram_outcome.details,
        ),
        CheckResult(
            key="lines",
            title="2. 逐行精确复制",
            passed=line_outcome.passed,
            metric=line_outcome.metric,
            threshold=line_outcome.threshold,
            details=line_outcome.details,
        ),
        CheckResult(
            key="strings",
            title=f"3. 字符串字面量相似度（阈值 {string_threshold:.2f}）",
            passed=string_outcome.passed,
            metric=string_outcome.metric,
            threshold=string_outcome.threshold,
            details=string_outcome.details,
        ),
        CheckResult(
            key="deps",
            title="4. 依赖审计",
            passed=dependency_outcome.passed,
            metric=dependency_outcome.metric,
            threshold=dependency_outcome.threshold,
            details=dependency_outcome.details,
        ),
        CheckResult(
            key="assets",
            title=f"5. 资产 / 文档审计（文档 n={doc_ngram}）",
            passed=asset_outcome.passed,
            metric=asset_outcome.metric,
            threshold=asset_outcome.threshold,
            details=asset_outcome.details,
        ),
    ]

    return AuditResult(
        target=target,
        reference=reference,
        head_commit=head_commit,
        head_subject=head_subject,
        ngram=ngram,
        doc_ngram=doc_ngram,
        fail_threshold=fail_threshold,
        file_threshold=file_threshold,
        string_threshold=string_threshold,
        min_matched_chars=min_matched_chars,
        max_copied_lines=max_copied_lines,
        min_copied_block=min_copied_block,
        generated_at=_iso_now(),
        reference_files=len(reference_files),
        reference_skipped=skipped,
        reference_source=source,
        target_files=len(target_files),
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------------- #

INTERPRETATION_SECTION = """## 如何解读本报告

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
"""


def render_report(result: AuditResult) -> str:
    """把审计结果渲染成 Markdown 报告。"""
    lines: list[str] = []
    overall = "✅ 全部通过" if result.passed else "❌ 存在失败项"

    lines.append("# 洁净室相似度审计报告（Clean-room Similarity Audit）")
    lines.append("")
    lines.append(
        "> 本报告由 `tools/similarity_audit.py` 自动生成，仅依据文本与结构证据判定，"
        "不构成法律意见。"
    )
    lines.append("")
    lines.append("## 元信息")
    lines.append("")
    lines.extend(
        _md_table(
            ["项目", "值"],
            [
                ("生成时间", result.generated_at),
                ("目标项目（待审计）", _md_code(str(result.target))),
                ("参考实现（比对基准）", _md_code(str(result.reference))),
                ("上游 HEAD commit", _md_code(result.head_commit)),
                ("上游 HEAD 提交信息", _md_cell(result.head_subject) or "（无）"),
                ("参考侧文件集来源", result.reference_source),
                ("上游 HEAD 跟踪文件数", f"{result.reference_files:,}（另有 {result.reference_skipped} 个路径在 HEAD 中不存在，已跳过）"),
                ("目标侧纳入审计的文件数", f"{result.target_files:,}"),
                ("n-gram 长度（源码 / 文档）", f"{result.ngram} / {result.doc_ngram}"),
                ("整体 containment 阈值", f"{result.fail_threshold:.2%}"),
                ("单文件 containment 阈值", f"{result.file_threshold:.2%}"),
                ("字符串相似度阈值", f"{result.string_threshold:.2f}"),
                ("字符串违规所需匹配字符数", str(result.min_matched_chars)),
                ("逐字复制：实质性行上限 / 对齐片段下限", f"{result.max_copied_lines} / {result.min_copied_block}"),
                ("审计结论", overall),
            ],
        )
    )
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.extend(
        _md_table(
            ["检查项", "状态", "关键指标", "阈值"],
            [
                (check.title, "✅ 通过" if check.passed else "❌ 失败", _md_cell(check.metric), _md_cell(check.threshold))
                for check in result.checks
            ],
        )
    )
    lines.append("")
    for check in result.checks:
        lines.append(f"## {check.title}")
        lines.append("")
        lines.append(f"**状态：** {'✅ 通过' if check.passed else '❌ 失败'}　　**关键指标：** {check.metric}　　**阈值：** {check.threshold}")
        lines.append("")
        lines.extend(check.details)
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(INTERPRETATION_SECTION)
    lines.append("")
    lines.append("## 方法与排除项")
    lines.append("")
    lines.extend(
        [
            "- **参考侧只读 HEAD**：文件列表由 `git -C <reference> ls-files` 得到，内容由 "
            "`git -C <reference> show HEAD:<path>` 读取；工作区中的未跟踪文件与本地改动"
            "（例如上游仓库里用户自己新增的笔记、`my_papers/`）一律不参与比对。",
            "- **目标侧遍历排除**：`.git/`、`__pycache__/`、`.venv/`、`node_modules/`、`build/`、"
            "`dist/`、`*.egg-info/`、审计报告自身，以及 `--exclude` 指定的模式。",
            "- **分词**：`.py` 使用保留注释与字符串字面量的正则分词器；`.md`/散文使用词/字级分词器。",
            f"- **行级比对**：仅比较 `.py`；忽略去空白后长度 < {MIN_LINE_LENGTH} 字符的行。命中行分三档："
            "`样板`（import / `if __name__` 等，完全忽略）、`通用惯用式`（类型守卫、异常捕获、"
            "字段声明、关键字实参行、调用头等，计数展示但不判失败）、`实质性`（判失败）。"
            "此外还检测**行号同步递增的对齐片段**：若目标第 t 行 = 上游第 r 行且 t+1 = r+1……"
            "连续出现，即使每行都很短也视为整块搬运。",
            f"- **字符串比对**：仅比较长度 ≥ {MIN_STRING_LENGTH} 字符、且非 docstring 的字面量；"
            "docstring 已由检查 1 覆盖，避免重复计数。判违规需同时满足"
            f"「相似度 ≥ {result.string_threshold:.2f}」与「实际匹配字符数 ≥ {result.min_matched_chars}」，"
            "后者用于排除 DOI / URL / 作者名 / 纯标识符等天然相似的事实性短串。",
            "- **专有标识符**：出现在注释 / docstring 中的归属声明或说明性提及记为警告，"
            "出现在代码或普通字符串中的记为违规（`--strict-identifiers` 可切换为全部判失败）。",
            "- **依赖审计**：审计工具自身文件（文件名含 `similarity_audit` 者、`tools/README.md`）"
            "会被排除，因为它们必然包含禁用词表。",
            "- **资产哈希**：跳过 0 字节文件（空文件哈希相同不具信息量）。",
            "- **未经检验的维度**：语义相似度、算法/架构相似度、Prompt 的改写式抄袭、"
            "由上游代码转译（例如 Python→其它语言）后的抄袭，均不在本脚本能力范围内。",
        ]
    )
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python tools/similarity_audit.py --target {result.target} --reference {result.reference} "
        f"--report docs/AUDIT.md --ngram {result.ngram} --fail-threshold {result.fail_threshold}"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise AuditError(f"无法写入报告 {path}：{exc}") from exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="similarity_audit.py",
        description="洁净室重写项目的反抄袭 / 相似度审计（仅使用标准库）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", required=True, help="待审计的新项目根目录")
    parser.add_argument("--reference", required=True, help="上游参考实现（git 仓库）根目录")
    parser.add_argument("--report", default="docs/AUDIT.md", help="Markdown 报告输出路径（相对当前工作目录）")
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="源码 token n-gram 长度")
    parser.add_argument("--doc-ngram", type=int, default=DEFAULT_DOC_NGRAM, help="文档 token n-gram 长度")
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=DEFAULT_FAIL_THRESHOLD,
        help="整体 containment 失败阈值（同时用于源码与文档）",
    )
    parser.add_argument(
        "--file-threshold",
        type=float,
        default=DEFAULT_FILE_THRESHOLD,
        help="单文件 containment 失败阈值",
    )
    parser.add_argument(
        "--string-threshold",
        type=float,
        default=DEFAULT_STRING_THRESHOLD,
        help="字符串字面量相似度失败阈值",
    )
    parser.add_argument(
        "--min-matched-chars",
        type=int,
        default=DEFAULT_MIN_MATCHED_CHARS,
        help="判定字符串违规所需的实际匹配字符数下限",
    )
    parser.add_argument(
        "--max-copied-lines",
        type=int,
        default=DEFAULT_MAX_COPIED_LINES,
        help="允许的逐字相同实质性代码行数上限",
    )
    parser.add_argument(
        "--min-copied-block",
        type=int,
        default=DEFAULT_MIN_COPIED_BLOCK,
        help="判定为整块搬运的“行号同步递增”对齐片段最小长度",
    )
    parser.add_argument(
        "--strict-identifiers",
        action="store_true",
        help="把注释 / docstring 中的上游专有标识符提及也计为违规（最严格口径）",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="报告中 Top-N 列表长度")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="额外排除的目标侧文件（glob，可重复）",
    )
    return parser


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 特殊流
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    report_path = Path(args.report).expanduser()
    if not report_path.is_absolute():
        report_path = (Path.cwd() / report_path).resolve()

    if args.ngram < 2 or args.doc_ngram < 2:
        print("错误：--ngram / --doc-ngram 必须 ≥ 2", file=sys.stderr)
        return 2
    if not 0.0 <= args.fail_threshold <= 1.0 or not 0.0 <= args.file_threshold <= 1.0:
        print("错误：阈值必须落在 [0, 1] 区间", file=sys.stderr)
        return 2
    if not 0.0 <= args.string_threshold <= 1.0:
        print("错误：--string-threshold 必须落在 [0, 1] 区间", file=sys.stderr)
        return 2

    try:
        result = run_audit(
            target,
            reference,
            report_path=report_path,
            ngram=args.ngram,
            doc_ngram=args.doc_ngram,
            fail_threshold=args.fail_threshold,
            file_threshold=args.file_threshold,
            string_threshold=args.string_threshold,
            min_matched_chars=args.min_matched_chars,
            max_copied_lines=args.max_copied_lines,
            min_copied_block=args.min_copied_block,
            top_n=args.top,
            excludes=args.exclude,
            strict_identifiers=args.strict_identifiers,
        )
        write_report(report_path, render_report(result))
    except AuditError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:  # pragma: no cover - 文件系统异常
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"目标项目：{result.target}")
    print(f"参考实现：{result.reference} @ {result.head_commit[:12]}（跟踪文件 {result.reference_files} 个）")
    print(f"目标文件：{result.target_files} 个")
    print("-" * 72)
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.title} :: {check.metric}")
    print("-" * 72)
    print(f"报告已写入：{report_path}")
    print("审计结论：" + ("全部通过 ✅" if result.passed else "存在失败项 ❌"))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

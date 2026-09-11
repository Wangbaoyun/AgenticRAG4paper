"""稳定哈希与确定性键派生。

**核心约束**：本模块的所有键派生必须是*确定性*的——同一语义输入在任何进程、
任何运行、任何机器上都必须产生同一个键。SPEC 中的 `SourceKey`、`FragmentId`、
`EvidenceKey`、`IndexFingerprint` 全部依赖此性质，它是"引用可复现"与
"索引可复用"两项质量目标的基础。

由此带来两条实现纪律：

1. 拼接多个片段时必须使用**不会出现在片段内部**的分隔符（这里用 ASCII Unit Separator
   ``\\x1f``），否则 ``("ab", "c")`` 与 ``("a", "bc")`` 会派生同一个键。
2. 参与哈希的文本先经 :func:`normalize_text` 做 Unicode NFKC 归一化与空白规整，
   否则同一份 PDF 在不同解析器下产生的不可见差异（全角/半角、NBSP、CRLF）会导致
   索引被无意义地重建。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

__all__ = ["derive_key", "hash_file", "normalize_text", "sha256_hex", "stable_json"]

#: 派生键时用于连接多个片段的单位分隔符（ASCII US，不会出现在正常文本中）。
_SEP = "\x1f"

_WHITESPACE_RE = re.compile(r"[^\S\n]+")  # 非换行的空白串（含 NBSP 归一化后的普通空格）
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def sha256_hex(data: str | bytes, *, length: int = 16) -> str:
    """返回 ``data`` 的 SHA-256 十六进制摘要，截断到 ``length`` 个字符。

    Args:
        data: 待哈希的文本（按 UTF-8 编码）或原始字节。
        length: 截断长度。16 个 hex 字符 = 64 bit，对本项目的规模足够避免碰撞。

    Returns:
        小写十六进制字符串。
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    if length <= 0 or length > 64:
        raise ValueError(f"length 必须在 1..64 之间，得到 {length}")
    return hashlib.sha256(data).hexdigest()[:length]


def stable_json(obj: Any) -> str:
    """把对象序列化为**规范化** JSON，用于派生确定性指纹。

    规范化的三要素：键排序（``sort_keys``）、紧凑分隔符（无多余空格）、
    保留非 ASCII（``ensure_ascii=False``，避免同一中文标题产生两种编码形态）。

    ``default=str`` 让 datetime / Path 等非 JSON 类型退化为字符串而不是抛异常——
    指纹计算不应因配置里多了一个 Path 字段而失败。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def derive_key(*parts: str, prefix: str = "", length: int = 16) -> str:
    """从若干文本片段派生一个确定性键。

    Args:
        *parts: 参与派生的片段，顺序敏感。
        prefix: 结果前缀，用于可读性与可 grep 性（如 ``"ev-"``）。
        length: 摘要截断长度。

    Returns:
        形如 ``f"{prefix}{hex}"`` 的字符串。

    Examples:
        >>> derive_key("a", "bc") != derive_key("ab", "c")
        True
    """
    payload = _SEP.join(normalize_text(part) for part in parts)
    return f"{prefix}{sha256_hex(payload, length=length)}"


def hash_file(path: Path, *, chunk_size: int = 1 << 20, length: int = 16) -> str:
    """流式计算文件内容的 SHA-256（截断）。

    使用流式读取，使得对超大 PDF 的哈希不会把整个文件读进内存
    （SPEC §3.10 "超长文档峰值内存不得随页数线性增长" 的配套要求）。

    Raises:
        FileNotFoundError: 路径不存在或不是文件。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def normalize_text(text: str) -> str:
    """Unicode NFKC 归一化 + 空白规整。

    处理三类在实际 PDF 中高频出现、但语义等价的差异：

    - 全角/半角与兼容字符（NFKC：``Ａ`` → ``A``，``⑴`` → ``(1)``）；
    - 不换行空格 NBSP(U+00A0)、零宽空格 ZWSP(U+200B) 等不可见字符；
    - 连续空格与三个以上连续换行。

    注意：**不做**大小写折叠，也不删除标点——那会改变文本语义，
    应当由调用方在需要时自行处理。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()

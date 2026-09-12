"""Unicode 清洗：保证**任何**字符串都能被 UTF-8 编码。

## 为什么需要独立一层

真实语料暴露过一次崩溃：PDF 字体编码表残缺时，pypdf 会把某些字形映射成
**孤立代理项**（U+D800–U+DFFF，例如 ``\\ud835``）。它们不是合法字符，
UTF-8 编码会抛 ``UnicodeEncodeError``。

当时的修法是"在 ``normalize_text`` 里清洗"，但那只覆盖了**恰好崩掉的那条路径**
（``Fragment.text``）。事后核查发现 ``Source.authors`` / ``venue`` / ``abstract`` /
``rel_path`` 仍可携带代理项，而这样的 ``Source`` 写 ``sources.jsonl`` 时
会让 ``model_dump_json()`` 抛 ``PydanticSerializationError``。

**根本问题不是"某个字段漏了清洗"，而是缺少一条"字符串离开进程前必然可编码"的保证。**
逐个字段补校验器会在下次新增字段时重演同一个事故。因此这里提供两个无法绕过的边界：

1. :class:`~scitrace.util.sanitize.SanitizedModel` —— 所有持久化模型的基类，
   用通配校验器覆盖**包括将来新增的**每一个字段；
2. :func:`sanitize_unicode` —— 存储层在序列化前的最后一道兜底。

## 清洗强度按字段区分（这一条搞混会引入更隐蔽的 bug）

- **正文类字段**：可以顺带做 NFKC 与空白规整（已有字段校验器在管）；
- **路径类字段**：**只清代理项，绝不做 NFKC**。NFKC 会改写路径中的字符，
  让它与磁盘上的真实文件对不上——那会静默破坏增量索引，
  表现为"文件明明在，却被当成新文件反复重建"。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["LONE_SURROGATE_RE", "sanitize_unicode", "sanitize_unicode_recursive"]

#: 孤立代理项。PDF 字体编码残缺与非法 UTF-8 文件名都会产生它们。
LONE_SURROGATE_RE = re.compile("[\ud800-\udfff]")

#: Unicode 替换字符。用它而不是删除：保留下来的"这里原本有个字符"这一信息，
#: 对排查"为什么这段文本缺了一块"比干净地删掉更有价值。
REPLACEMENT_CHARACTER = "\ufffd"


def sanitize_unicode(value: Any) -> Any:
    """把任意结构中的孤立代理项替换为 U+FFFD。

    递归处理 ``str`` / ``list`` / ``tuple`` / ``dict``（键与值都处理），
    其它类型原样返回。**不处理嵌套模型**——模型自带校验器，见
    :class:`SanitizedModel`。

    之所以要递归：pydantic 的通配校验器只会收到**当前模型这一层**的原始值，
    而 ``authors`` 是 ``list[str]``、``tags`` 可能是 ``dict[str, str]``，
    只处理顶层 ``str`` 会漏掉它们。

    Args:
        value: 任意值。

    Returns:
        清洗后的值；输入不含代理项时返回等价结构。
    """
    if isinstance(value, str):
        return LONE_SURROGATE_RE.sub(REPLACEMENT_CHARACTER, value) if LONE_SURROGATE_RE.search(value) else value
    if isinstance(value, list):
        return [sanitize_unicode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_unicode(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_unicode(key): sanitize_unicode(item) for key, item in value.items()
        }
    return value


def contains_lone_surrogate(value: Any) -> bool:
    """结构中是否含孤立代理项。供测试与排障使用。"""
    if isinstance(value, str):
        return bool(LONE_SURROGATE_RE.search(value))
    if isinstance(value, list | tuple):
        return any(contains_lone_surrogate(item) for item in value)
    if isinstance(value, dict):
        return any(
            contains_lone_surrogate(key) or contains_lone_surrogate(item)
            for key, item in value.items()
        )
    return False


#: 供类型标注与文档引用；实际实现是同一个函数。
sanitize_unicode_recursive = sanitize_unicode


def make_sanitized_base() -> type:
    """构造 :class:`SanitizedModel` 所需的基类。

    单独一个工厂函数是为了避免 ``util`` 依赖 ``pydantic``：
    ``util`` 是纯标准库层，而本模块的其余部分（:func:`sanitize_unicode`）
    也确实不需要 pydantic。只有模型基类需要，因此把它隔离在这里。

    实际使用请直接 ``from scitrace.util.sanitize import SanitizedModel``。
    """
    from pydantic import BaseModel, ConfigDict, field_validator  # noqa: PLC0415

    class SanitizedModel(BaseModel):
        """所有会被持久化的模型的基类。

        用 ``@field_validator("*", mode="before")`` 覆盖**每一个**字段——
        包括将来新增的。这一点是刻意的：这次事故的根因就是"漏了一个字段"，
        而逐个字段补校验器的方案无法阻止下一次遗漏。

        ``mode="before"`` 让清洗发生在类型转换**之前**，因此即使字段类型是
        ``list[str]`` 或 ``dict[str, str]`` 也能被递归覆盖。
        """

        model_config = ConfigDict(extra="forbid")

        @field_validator("*", mode="before")
        @classmethod
        def _sanitize_unicode(cls, value: Any) -> Any:
            return sanitize_unicode(value)

    return SanitizedModel


#: 所有持久化模型的基类。**不要在 util 之外重新定义它**——
#: 两套基类意味着两套清洗规则，而"规则不一致"正是这类问题的温床。
SanitizedModel = make_sanitized_base()

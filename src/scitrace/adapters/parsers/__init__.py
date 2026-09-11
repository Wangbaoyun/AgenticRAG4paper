"""解析器注册表：按名字或按文件选择解析后端。

上层（``pipeline.ingest``、``config``）只与本模块打交道，不直接 import
具体的解析器类。因此"新增一个解析后端"等于"在这里注册一行"。

解析器名会进入索引指纹（SPEC §5.3），所以名字必须是稳定的字符串常量，
不能随类名重构而漂移。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scitrace.adapters.parsers.plaintext import PlainTextParser
from scitrace.adapters.parsers.pypdf_parser import PyPDFParser
from scitrace.ports import DocumentParser

__all__ = [
    "available_parsers",
    "get_parser",
    "register_parser",
    "select_parser",
]


#: 名字 → 工厂。用工厂而非实例：解析器可能持有状态（连接、线程池），
#: 且测试需要拿到干净实例。
_REGISTRY: dict[str, Callable[[], DocumentParser]] = {
    PlainTextParser.name: PlainTextParser,
    PyPDFParser.name: PyPDFParser,
}


def register_parser(name: str, factory: Callable[[], DocumentParser]) -> None:
    """注册一个解析器实现。

    Args:
        name: 解析器名。会写入索引指纹，**一旦发布不应更改**；
            改名等同于换解析器，会让既有索引失效。
        factory: 无参工厂，返回满足 :class:`~scitrace.ports.DocumentParser` 的对象。
    """
    _REGISTRY[name] = factory


def available_parsers() -> list[str]:
    """返回已注册的解析器名（排序后）。"""
    return sorted(_REGISTRY)


def get_parser(name: str) -> DocumentParser:
    """按名字获取解析器实例。

    Raises:
        ValueError: 名字未注册。错误信息里会列出全部可用名字——
            配置项写错一个字母却只得到"解析器不存在"，用户无从查起。
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"未知的解析器 {name!r}；可用解析器：{', '.join(available_parsers())}"
        )
    return factory()


def select_parser(path: Path, *, preferred: str | None = None) -> DocumentParser:
    """为给定文件选择解析器。

    选择顺序：

    1. ``preferred`` 指定的解析器，若它 ``supports`` 该文件；
    2. 否则按注册顺序找一个 ``supports`` 该文件的解析器；
    3. 都不支持 → 抛 :class:`ValueError`。

    刻意**不**在 ``preferred`` 不支持时静默回退到别的解析器：那会让
    "配置了 pymupdf 但实际用了 pypdf"这种错误永远不被发现，
    而两者的输出差异会直接改变索引内容。

    Args:
        path: 待解析文件。
        preferred: 配置中指定的解析器名；``None`` 表示自动选择。

    Raises:
        ValueError: 找不到能处理该文件的解析器，或 ``preferred`` 不支持该文件。
    """
    if preferred is not None:
        parser = get_parser(preferred)
        if parser.supports(path):
            return parser
        raise ValueError(
            f"解析器 {preferred!r} 不支持文件 {path.name}。"
            f"可用解析器：{', '.join(available_parsers())}"
        )

    for name in available_parsers():
        parser = _REGISTRY[name]()
        if parser.supports(path):
            return parser

    raise ValueError(
        f"没有解析器能处理 {path.name}（后缀 {path.suffix or '无'}）。"
        f"可用解析器：{', '.join(available_parsers())}"
    )

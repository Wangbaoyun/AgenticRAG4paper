"""文档解析器契约。

解析器是整条链路的**入口**，也是错误最集中的地方：PDF 千奇百怪，扫描件、
加密文件、损坏字体、双栏排版都会出现。因此契约把"失败"设计成一等公民——
:class:`ParseError` 是预期内的结果，而不是意外。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from scitrace.domain import ParsedDocument

__all__ = ["DocumentParser", "ParseError"]


class ParseError(RuntimeError):
    """文档解析失败。

    上层（``pipeline.ingest``）捕获此异常后把该文件标记为 ``status=failed``
    并继续处理其余文件——**单个坏文件不得中断整批摄入**（SPEC §3.1）。
    """


@runtime_checkable
class DocumentParser(Protocol):
    """把文件转换为按页组织的文本。

    实现方的职责边界（SPEC §3.1）：

    - **只做解析**：不切块、不补元数据、不碰索引；
    - **不抛未包装的异常**：底层库的错误应转为 :class:`ParseError`，
      并携带足以定位问题的信息（文件路径 + 原始错误）；
    - **``name`` 参与索引指纹**：更换解析器必须导致索引重建，因为
      同一份 PDF 在不同解析器下的文本不同。
    """

    @property
    def name(self) -> str:
        """解析器标识（如 ``"pypdf"``）。参与索引指纹计算（SPEC §5.3）。"""
        ...

    def supports(self, path: Path) -> bool:
        """该解析器是否能处理此文件。

        通常按后缀判断；``.pdf`` 之外还应支持 ``.txt`` / ``.md`` 等纯文本格式。
        """
        ...

    async def parse(self, path: Path) -> ParsedDocument:
        """解析文件。

        Args:
            path: 待解析文件的路径。

        Returns:
            按页组织的文本与内嵌元数据线索。

        Raises:
            ParseError: 文件不存在、格式不支持、内容损坏或加密。

        Note:
            实现应为 **async**：解析是 CPU/IO 密集操作，同步实现会阻塞事件循环，
            使并发摄入与并发元数据补全退化为串行。同步库（如 pypdf）的调用
            请用 ``anyio.to_thread.run_sync`` 包一层。
        """
        ...

"""PDF 解析器（基于 pypdf）。

职责边界（SPEC §3.1）：**只把 PDF 变成按页组织的文本**，不做分块、不补元数据、
不碰索引。换一个解析后端（PyMuPDF、Docling…）只需另实现本协议，
其余环节一行都不用改。

## 为什么明确区分"可恢复失败"与"不可恢复失败"

真实语料里的 PDF 有相当比例是有问题的：扫描件（无文本层）、加密、单页损坏、
字体缺字。这些情况的正确处置各不相同：

- **单页损坏** → 记录并跳过该页，其余页照常产出（部分总好过全无）；
- **整体加密 / 无文本层** → 抛 :class:`ParseError`，由摄入层标记 ``status=failed``，
  不阻塞同批其他文件。

把两者都当成异常抛出，会让一批 100 篇文献因为 1 篇扫描件而中断；
都当成静默跳过，则用户永远不知道自己的语料里有扫描件。
"""

from __future__ import annotations

import logging
from pathlib import Path

import anyio

from scitrace.domain import ParsedDocument, ParsedPage
from scitrace.ports import ParseError

logger = logging.getLogger(__name__)

__all__ = ["PyPDFParser"]


class PyPDFParser:
    """使用 ``pypdf`` 抽取 PDF 文本。"""

    #: 参与索引指纹的解析器名（SPEC §5.3）。更换解析器必须导致索引重建，
    #: 因为同一份 PDF 在不同解析器下的文本不同。
    name = "pypdf"

    SUFFIXES: frozenset[str] = frozenset({".pdf"})

    #: 单页抽取出的文本少于该字符数时，视为"无文本层"信号。
    MIN_TEXT_PER_PAGE = 20

    def supports(self, path: Path) -> bool:
        """是否支持该文件。"""
        return path.suffix.lower() in self.SUFFIXES

    async def parse(self, path: Path) -> ParsedDocument:
        """解析 PDF。

        Note:
            ``pypdf`` 是纯 CPU 的同步库，这里用 ``anyio.to_thread.run_sync`` 包一层，
            否则解析一本大部头会阻塞事件循环，使并发摄入退化为串行。

        Raises:
            ParseError: 文件不存在、加密、损坏，或整份文档不含可抽取文本。
        """
        return await anyio.to_thread.run_sync(self._parse_sync, Path(path))

    def _parse_sync(self, path: Path) -> ParsedDocument:  # noqa: PLR0912
        try:
            from pypdf import PdfReader
        except ImportError as error:  # pragma: no cover - 依赖缺失时的明确指引
            raise ParseError(
                "未安装 pypdf。请执行 `pip install scitrace[pypdf]` 或改用其他解析器。"
            ) from error

        if not path.is_file():
            raise ParseError(f"文件不存在：{path}")

        try:
            reader = PdfReader(str(path))
        except Exception as error:  # noqa: BLE001 — pypdf 的异常类型不稳定
            raise ParseError(f"无法打开 PDF {path.name}：{error}") from error

        if reader.is_encrypted:
            # 尝试空密码解密（部分 PDF 只是设置了口令位但未真正加密）
            try:
                if reader.decrypt("") == 0:
                    raise ParseError(f"PDF 已加密且无法用空密码解密：{path.name}")
            except ParseError:
                raise
            except Exception as error:  # noqa: BLE001
                raise ParseError(f"PDF 已加密：{path.name}（{error}）") from error

        pages: list[ParsedPage] = []
        failed_pages = 0
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as error:  # noqa: BLE001 — 单页损坏不应毁掉整份文档
                failed_pages += 1
                logger.warning("%s 第 %d 页抽取失败，已跳过：%s", path.name, index, error)
                continue
            pages.append(ParsedPage(page_number=index, text=text))

        total_chars = sum(len(page.text.strip()) for page in pages)
        page_count = len(reader.pages)
        if page_count and total_chars < self.MIN_TEXT_PER_PAGE * max(1, len(pages)):
            raise ParseError(
                f"PDF 不含可抽取文本层（疑似扫描件）：{path.name}；"
                "如需处理扫描件，请接入 OCR 型解析后端。"
            )
        if not pages:
            raise ParseError(f"PDF 未产出任何页面：{path.name}")

        hints = self._extract_hints(reader, path)
        if failed_pages:
            logger.warning("%s 共有 %d 页抽取失败", path.name, failed_pages)

        logger.info("解析完成：%s（%d 页，%d 字符）", path.name, len(pages), total_chars)
        return ParsedDocument(pages=pages, hints=hints, parser=self.name)

    @staticmethod
    def _extract_hints(reader: object, path: Path) -> dict[str, str]:
        """从 PDF 内嵌元数据与文件名中提取候选书目线索。

        这些只是**线索**，会作为元数据补全的起点，但不会被直接当作权威书目数据：
        PDF 内嵌元数据由排版软件写入，错误率相当高（常见的是把标题写成
        "Microsoft Word - 未命名文档.docx"）。
        """
        hints: dict[str, str] = {}
        metadata = getattr(reader, "metadata", None)
        if metadata:
            for source_key, target_key in (
                ("/Title", "title"),
                ("/Author", "authors"),
                ("/CreationDate", "date"),
                ("/Subject", "subject"),
            ):
                value = metadata.get(source_key)
                if not isinstance(value, str) or not value.strip():
                    continue
                cleaned = value.strip()
                # 排版软件写进去的占位标题（"Microsoft Word - 未命名文档.docx"）
                # 比没有标题更糟：它会被当作线索送进元数据补全，污染检索结果。
                if target_key == "title" and _is_placeholder_title(cleaned, path):
                    logger.debug("忽略占位标题：%r", cleaned)
                    continue
                hints[target_key] = cleaned
        # **文件名只作为最后兜底**，不进 ``title``。
        # 它混淆进 title 的后果在真实数据上立刻显现：本项目的样例论文
        # 内嵌标题是 "paperqa2"，而 ``apply_patch`` 又规定"不覆盖已有值"，
        # 于是 Crossref 查到的真实标题永远补不上，文内引用渲染成
        # ``(anonndpaperqa2 pages 2-3)``。
        hints["fallback_title"] = path.stem
        hints["filename"] = path.name
        return hints


def _is_placeholder_title(title: str, path: Path | None = None) -> bool:
    """判断内嵌标题是否是排版软件留下的占位符。

    多一条判据：**标题与文件名相同**。很多工具在导出 PDF 时把文件名写进
    Title 字段，这样的标题不携带任何信息，却会被当作"线索"送去元数据补全，
    而且因为合并规则不覆盖已有值，它还会**永久挡住**查到的真实标题。
    """
    lowered = title.strip().lower()
    if (
        lowered.endswith((".doc", ".docx", ".tex", ".indd"))
        or lowered in {"untitled", "unknown", "document", "microsoft word"}
        or "untitled" in lowered
    ):
        return True
    return path is not None and lowered == path.stem.strip().lower()

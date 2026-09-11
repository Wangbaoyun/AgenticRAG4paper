"""撤稿状态来源：本地快照。

撤稿检查**刻意不做成在线查询**，有两个具体理由：

1. **它必须可靠**。把一篇已撤稿论文的结论当作有效证据引用，是这类系统最严重的
   失效模式——比"检索不到"严重得多，因为用户不会察觉。依赖一个可能超时、
   可能限流的在线端点，等于把最关键的判断交给了最不可靠的环节；
2. **它变化很慢**。撤稿是低频事件，一份定期更新的本地快照完全够用，
   而在线逐篇查询会把每次摄入的延迟成倍放大。

因此本来源读一份 CSV 快照（默认兼容 Retraction Watch 数据库的导出列名），
建一次内存索引后按 DOI 命中。未配置快照时它安静地不贡献任何字段——
这比"查不到就当作未撤稿"更诚实，因为后者是**把未知当成已知**。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from scitrace.domain import SourcePatch
from scitrace.domain.source import normalize_doi
from scitrace.ports import MetadataMatch, MetadataProvider

logger = logging.getLogger(__name__)

__all__ = ["RetractionProvider"]

#: 候选的 DOI 列名。Retraction Watch 导出用 ``OriginalPaperDOI``，
#: 其他快照常见 ``doi`` / ``DOI`` / ``original_doi``，一并接受。
_DOI_COLUMNS: tuple[str, ...] = (
    "OriginalPaperDOI",
    "original_paper_doi",
    "original_doi",
    "doi",
    "DOI",
)

#: 候选的标题列名，仅用于日志与排障。
_TITLE_COLUMNS: tuple[str, ...] = ("Title", "title", "RecordTitle")


class RetractionProvider:
    """按 DOI 在本地快照中查找撤稿记录。"""

    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None
        self._retracted_dois: dict[str, str] = {}
        self._loaded = False
        self._load_error: str | None = None

    @property
    def name(self) -> str:
        return "retraction"

    @property
    def size(self) -> int:
        """快照中的撤稿记录数（加载后可用）。"""
        self._ensure_loaded()
        return len(self._retracted_dois)

    def _ensure_loaded(self) -> None:
        """惰性加载快照。

        加载失败只记录一次警告并降级为空索引——**不抛异常**，
        因为一个损坏的快照不该让整个摄入失败，但也不能静默：
        用户需要知道"这次没有做撤稿检查"。
        """
        if self._loaded:
            return
        self._loaded = True
        if self.csv_path is None:
            logger.info("未配置撤稿快照，本次摄入不做撤稿检查")
            return
        if not self.csv_path.is_file():
            self._load_error = f"文件不存在：{self.csv_path}"
            logger.warning("撤稿快照%s，已跳过撤稿检查", self._load_error)
            return
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                doi_column = _pick_column(reader.fieldnames or [], _DOI_COLUMNS)
                if doi_column is None:
                    self._load_error = f"找不到 DOI 列（表头：{reader.fieldnames}）"
                    logger.warning("撤稿快照%s，已跳过撤稿检查", self._load_error)
                    return
                title_column = _pick_column(reader.fieldnames or [], _TITLE_COLUMNS)
                for row in reader:
                    doi = normalize_doi(row.get(doi_column))
                    if doi:
                        self._retracted_dois[doi] = (row.get(title_column) or "") if title_column else ""
        except Exception as error:  # noqa: BLE001 - 快照损坏不应中断摄入
            self._load_error = str(error)
            logger.warning("撤稿快照读取失败（%s），已跳过撤稿检查", error)
            return
        logger.info("撤稿快照加载完成：%d 条记录", len(self._retracted_dois))

    async def lookup(self, patch: SourcePatch) -> MetadataMatch | None:
        """若该文献的 DOI 在撤稿快照中，返回 ``retracted=True``。"""
        self._ensure_loaded()
        doi = normalize_doi(patch.doi)
        if not doi or doi not in self._retracted_dois:
            return None
        title = self._retracted_dois[doi]
        logger.warning("文献已被撤稿：%s（%s）", doi, title or "标题未知")
        return MetadataMatch(
            patch=SourcePatch(retracted=True),
            provider=self.name,
            confidence="doi",
            matched_title=title or None,
            score=1.0,
        )

    async def aclose(self) -> None:
        """无外部资源需要释放。"""
        return None


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    """在表头中找出第一个匹配的列名（大小写不敏感）。"""
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None

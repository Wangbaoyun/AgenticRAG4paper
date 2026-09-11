"""文献元数据与会话记录的本地存储。

## 设计取舍：为什么是 JSONL + 原子替换，而不是 SQLite

候选方案有三种：SQLite、逐条 JSON 文件、单个 JSONL 文件。选择最后一个的理由：

- **规模**：一个语料通常几百到几千篇文献。几千条记录的读写用 SQLite 是杀鸡用牛刀，
  而它带来的迁移/版本管理成本是实打实的；
- **可读性**：``sources.jsonl`` 可以被 ``grep``、被 ``jq``、被直接打开看。
  排查"这条引用为什么渲染错了"时，能直接看到那条记录，比写 SQL 快得多；
- **崩溃安全**：写入走"临时文件 + ``replace``"，POSIX 保证替换是原子的。
  最坏情况是丢掉最后一次写入（下次摄入会补上），而不是留下半截文件。

真正的规模瓶颈在成百上千篇文献的**片段**上，那由向量索引与全文索引承担，
和本模块无关。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from scitrace.domain import Session, Source

logger = logging.getLogger(__name__)

__all__ = ["SessionStore", "SourceStore"]

SOURCES_FILENAME = "sources.jsonl"


class _JsonlStore:
    """JSONL 存储的公共实现。

    子类只需给出文件名、模型类型与主键提取方式。
    """

    filename: str = ""
    label: str = "记录"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / self.filename

    # --- 子类实现 ---

    def _model_validate(self, payload: str):  # noqa: ANN202 - 由子类收窄
        raise NotImplementedError

    @staticmethod
    def _dump(record) -> str:  # noqa: ANN001 - 由子类收窄
        raise NotImplementedError

    @staticmethod
    def _key_of(record, fallback: str) -> str:  # noqa: ANN001
        raise NotImplementedError

    # --- 公共行为 ---

    def exists(self) -> bool:
        """存储文件是否已存在。"""
        return self.path.is_file()

    def load(self) -> dict[str, object]:
        """读取全部记录。

        单行损坏时**跳过该行并继续**，而不是让整个存储不可用：
        一份 5000 篇文献的元数据因为一行截断而完全读不出来，
        是远比丢失一条记录更严重的故障。
        """
        if not self.path.is_file():
            return {}
        records: dict[str, object] = {}
        malformed = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = self._model_validate(stripped)
                except Exception:  # noqa: BLE001 - 单行损坏不应毁掉整个存储
                    malformed += 1
                    continue
                records[self._key_of(record, str(line_number))] = record
        if malformed:
            logger.warning("%s 中有 %d 行无法解析，已跳过（%s）", self.label, malformed, self.path)
        return records

    def save(self, records: Iterable[object]) -> None:
        """原子写入全部记录。"""
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(self._dump(record))
                handle.write("\n")
        temporary.replace(self.path)

    def clear(self) -> None:
        """删除存储文件。"""
        self.path.unlink(missing_ok=True)


class SourceStore(_JsonlStore):
    """文献元数据存储：``source_key -> Source``。"""

    filename = SOURCES_FILENAME
    label = "文献元数据"

    def _model_validate(self, payload: str) -> Source:
        return Source.model_validate_json(payload)

    @staticmethod
    def _dump(record: Source) -> str:
        return record.model_dump_json()

    @staticmethod
    def _key_of(record: Source, fallback: str) -> str:
        return record.key or fallback

    # --- 类型收窄的便捷封装 ---

    def load_sources(self) -> dict[str, Source]:
        """读取全部文献元数据。"""
        return {key: value for key, value in self.load().items() if isinstance(value, Source)}

    def save_sources(self, sources: Mapping[str, Source]) -> None:
        """写入全部文献元数据，按键排序以保证输出确定可比对。"""
        self.save([sources[key] for key in sorted(sources)])


class SessionStore(_JsonlStore):
    """会话历史存储。

    与元数据不同，会话记录可能很多且会持续增长，因此按 ``session_id`` 直接
    覆写单条记录而不是每次重写整个文件。
    """

    filename = "sessions.jsonl"
    label = "会话记录"

    def _model_validate(self, payload: str) -> Session:
        return Session.model_validate_json(payload)

    @staticmethod
    def _dump(record: Session) -> str:
        return record.model_dump_json()

    @staticmethod
    def _key_of(record: Session, fallback: str) -> str:
        return record.session_id or fallback

    def load_sessions(self) -> dict[str, Session]:
        """读取全部会话记录。"""
        return {key: value for key, value in self.load().items() if isinstance(value, Session)}

    def append(self, session: Session) -> Path:
        """追加一条会话记录。

        同名 ``session_id`` 的旧记录不会被删除，读取时以最后一条为准——
        保留历史版本对复现实验有价值，而读取语义由 :meth:`load_sessions` 的
        "后写覆盖"保证。
        """
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json.loads(session.model_dump_json()), ensure_ascii=False))
            handle.write("\n")
        return self.path

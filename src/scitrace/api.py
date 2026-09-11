"""对外的高层 API：CLI 与库调用共用同一组入口。

存在的意义是**让 CLI 保持很薄**。命令行的参数解析、输出格式、退出码都是
容易变的东西；把业务编排放进 CLI 会让每次调整输出都触碰业务逻辑。
这里返回的是领域对象（``AgentRunResult`` / ``ScoredFragment`` / ``IngestReport``），
序列化与排版完全交给调用方。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from scitrace.agent.runtime import AgentRunResult, AgentRuntime
from scitrace.config import Settings
from scitrace.factory import build_ingest_pipeline, build_services
from scitrace.pipeline.ingest import IngestReport, ManifestStore
from scitrace.ports import ScoredFragment
from scitrace.service import SourceStore

logger = logging.getLogger(__name__)

__all__ = ["ask", "build_index", "index_status", "search"]


async def ask(
    question: str,
    services,  # noqa: ANN001 - factory.Services
    *,
    mode: str = "deterministic",
) -> AgentRunResult:
    """回答一个问题。

    Args:
        question: 用户问题。
        services: 已装配的服务集合。
        mode: ``deterministic`` 或 ``agentic``。

    Returns:
        运行结果。**拒答也是成功返回**——它由 ``result.status`` 表达，
        不是异常。
    """
    runtime = AgentRuntime(services=services, question=question, mode=mode)  # type: ignore[arg-type]
    return await runtime.run()


async def search(
    query: str, services, *, k: int | None = None  # noqa: ANN001
) -> list[ScoredFragment]:
    """只做检索，不筛选、不合成。用于查看"证据长什么样"。"""
    return await services.retriever.retrieve(query, k=k)


async def build_index(
    services, paths: Sequence[Path], *, rebuild: bool = False  # noqa: ANN001
) -> IngestReport:
    """摄入语料并建立索引。"""
    pipeline = build_ingest_pipeline(services.settings, services)
    report = await pipeline.run(list(paths), rebuild=rebuild)  # type: ignore[attr-defined]
    # 摄入会改写索引与元数据，而 `services` 里那份 `sources` 是构造时从磁盘读的快照。
    # 不刷新的话，"同一个进程内先摄入再提问"会拿着空/陈旧的元数据去渲染引用——
    # 表现为引用全是 `unknown`，而索引里其实有完整的书目信息。
    # CLI 每次命令重建 services 所以掩盖了这个陷阱；边界测试用同一进程跑就暴露了。
    services.sources = SourceStore(services.settings.index_dir).load_sources()
    return report


def index_status(settings: Settings) -> dict[str, object]:
    """报告当前配置对应的索引状态。

    **不抛异常**：索引不存在、清单损坏、元数据文件缺失都返回零值——
    一个"状态查询"命令因为索引不存在而报错，会让用户无法判断
    "是索引坏了还是我配错了"。
    """
    index_dir = settings.index_dir
    status: dict[str, object] = {
        "fingerprint": settings.index_fingerprint(),
        "index_dir": str(index_dir),
        "exists": index_dir.exists(),
        "sources": 0,
        "fragments": 0,
        "corpus": {},
        "failed": 0,
    }
    try:
        manifest = ManifestStore(index_dir).load()
    except Exception as error:  # noqa: BLE001 - 状态查询不应失败
        logger.warning("无法读取索引清单：%s", error)
        return status

    entries = manifest.entries
    status["fragments"] = sum(
        entry.fragment_count for entry in entries.values() if entry.status == "ok"
    )
    status["failed"] = sum(1 for entry in entries.values() if entry.status == "failed")

    # 语料统计按 rel_path 的**首段**聚合——摄入时标识形如 ``<根目录名>/<相对路径>``
    corpus: dict[str, int] = {}
    for identifier in entries:
        root = identifier.split("/", 1)[0] if "/" in identifier else "(单文件)"
        corpus[root] = corpus.get(root, 0) + 1
    status["corpus"] = corpus

    try:
        status["sources"] = len(SourceStore(index_dir).load_sources())
    except Exception as error:  # noqa: BLE001
        logger.warning("无法读取文献元数据：%s", error)
    return status


def load_services(settings: Settings, *, language: str = "zh", load_index: bool = True):  # noqa: ANN201
    """``build_services`` 的转发，便于 CLI 单点 monkeypatch。"""
    return build_services(settings, language=language, load_index=load_index)

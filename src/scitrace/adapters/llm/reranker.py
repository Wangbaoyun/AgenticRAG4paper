"""交叉编码器重排器（基于 sentence-transformers）。

与 :class:`~scitrace.adapters.llm.embedding.LocalEmbeddingClient` 的关键差别：
嵌入模型把查询与文档**分别**编码成向量（相似度是事后算的点积），
交叉编码器把 ``(查询, 文档)`` **一起**送进模型，因此能建模两者之间的交互——
精度明显更高，代价是无法预计算：每个候选都要跑一次前向。

这决定了它的用法：只对**已经收敛的候选池**（几十条）使用，
绝不用它做全库检索。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anyio

from scitrace.domain import Fragment
from scitrace.ports import ScoredFragment

logger = logging.getLogger(__name__)

__all__ = ["CrossEncoderReranker"]


class CrossEncoderReranker:
    """满足 :class:`~scitrace.ports.reranker.Reranker` 协议的本地实现。"""

    def __init__(self, model: str, *, batch_size: int = 16, device: str | None = None) -> None:
        self._model_name = model
        self._batch_size = max(1, batch_size)
        self._device = device
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> object:
        """惰性加载：仅构造配置对象不应触发模型下载。"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "未安装 sentence-transformers。请执行 `pip install scitrace[rerank]`，"
                "或把 retrieval.strategy 改为 hybrid_rrf。"
            ) from error
        logger.info("加载重排模型：%s", self._model_name)
        self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self, query: str, fragments: Sequence[Fragment], *, top_n: int
    ) -> list[ScoredFragment]:
        """对候选片段打分并返回前 ``top_n`` 条。"""
        if not fragments:
            return []
        pairs = [(query, item.text) for item in fragments]
        scores = await anyio.to_thread.run_sync(self._score_sync, pairs)

        ranked = sorted(
            zip(fragments, scores, strict=True), key=lambda pair: (-pair[1], pair[0].fragment_id)
        )
        return [
            ScoredFragment(fragment=item, score=float(score), origin="rerank", rank=position)
            for position, (item, score) in enumerate(ranked[: max(0, top_n)], start=1)
        ]

    def _score_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure_loaded()
        raw = model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)  # type: ignore[attr-defined]
        return [float(value) for value in raw]

    async def aclose(self) -> None:
        """释放模型句柄。"""
        self._model = None

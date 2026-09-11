"""嵌入客户端：本地模型与 OpenAI 兼容端点。

两种后端的定位不同，配置里各有取舍：

- **本地**（``sentence-transformers``）：中文场景默认。无需 API key、
  无网络延迟、数据不出本机；代价是首次加载模型要下载数百 MB 并占用显存/内存，
  且推理是 CPU/GPU 密集操作。
- **OpenAI 兼容端点**：适合已有自建嵌入服务的场景，或本地资源受限时。

两者都在**适配器边界**做 L2 归一化，把端口约定落到实处——见
:func:`scitrace.adapters.llm.convert.normalize_vector` 的说明。
"""

from __future__ import annotations

import logging
from typing import Literal

import anyio
import httpx

from scitrace.adapters.llm.convert import normalize_vector
from scitrace.adapters.llm.retry import RetryPolicy, with_retries

logger = logging.getLogger(__name__)

__all__ = ["LocalEmbeddingClient", "OpenAIEmbeddingClient"]


class LocalEmbeddingClient:
    """基于 ``sentence-transformers`` 的本地嵌入。

    模型**延迟加载**：构造配置对象（例如只为了打印 ``stc config show``）不应触发
    数百 MB 的模型下载。
    """

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 32,
        query_prefix: str = "",
        dimension: int | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name = model
        self._batch_size = max(1, batch_size)
        self._query_prefix = query_prefix
        self._dimension = dimension or 0
        self._device = device
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """向量维度。首次调用 :meth:`embed` 后才会被探测出来。"""
        return self._dimension

    def _ensure_loaded(self) -> object:
        """加载模型（同步，调用方须放进线程池）。"""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "未安装 sentence-transformers。请执行 "
                "`pip install scitrace[local-embedding]`，或把 embedding.backend 改为 openai。"
            ) from error
        logger.info("加载本地嵌入模型：%s", self._model_name)
        model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = int(model.get_sentence_embedding_dimension())
        self._model = model
        return model

    async def embed(
        self, texts: list[str], *, kind: Literal["query", "document"] = "document"
    ) -> list[list[float]]:
        """把文本编码为归一化向量。

        ``kind="query"`` 时会加上 ``query_prefix``：BGE 一类非对称模型要求查询侧
        带指令前缀，漏掉会让检索质量明显下降却不报任何错。
        """
        if not texts:
            return []
        prepared = (
            [f"{self._query_prefix}{text}" for text in texts]
            if kind == "query" and self._query_prefix
            else list(texts)
        )
        return await anyio.to_thread.run_sync(self._embed_sync, prepared)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=False,  # 归一化统一由 normalize_vector 负责，避免两套实现
            show_progress_bar=False,
        )
        result = [normalize_vector([float(value) for value in vector]) for vector in vectors]
        if self._dimension == 0 and result:
            self._dimension = len(result[0])
        return result


class OpenAIEmbeddingClient:
    """调用 OpenAI 兼容的 ``/v1/embeddings`` 端点。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        dimension: int | None = None,
        timeout_s: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model
        base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self._endpoint = f"{base}/embeddings"
        self._api_key = api_key
        self._batch_size = max(1, batch_size)
        self._query_prefix = query_prefix
        self._dimension = dimension or 0
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy or RetryPolicy()
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def embed(
        self, texts: list[str], *, kind: Literal["query", "document"] = "document"
    ) -> list[list[float]]:
        """分批调用端点并归一化结果。

        分批是必需的：多数提供商对单次请求的输入条数或总 token 有上限，
        超过会返回 400 而不是自动截断。
        """
        if not texts:
            return []
        prepared = (
            [f"{self._query_prefix}{text}" for text in texts]
            if kind == "query" and self._query_prefix
            else list(texts)
        )

        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._timeout_s)
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(prepared), self._batch_size):
                batch = prepared[start : start + self._batch_size]
                vectors.extend(await self._embed_batch(client, batch))
            if self._dimension == 0 and vectors:
                self._dimension = len(vectors[0])
            return vectors
        finally:
            if owns_client:
                await client.aclose()

    async def _embed_batch(self, client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
        async def call() -> httpx.Response:
            response = await client.post(
                self._endpoint,
                headers=self._headers(),
                json={"model": self._model_name, "input": batch},
            )
            response.raise_for_status()
            return response

        response = await with_retries(
            call, policy=self._retry_policy, description=f"嵌入调用（{self._model_name}）"
        )
        payload = response.json()
        rows = payload.get("data") or []
        if len(rows) != len(batch):
            raise RuntimeError(
                f"嵌入端点返回条数不匹配：期望 {len(batch)}，得到 {len(rows)}"
            )
        # 按 index 排序：并行响应顺序不保证与输入一致，不排序会静默把向量配错文本
        rows.sort(key=lambda row: row.get("index", 0))
        return [normalize_vector([float(value) for value in row["embedding"]]) for row in rows]

    async def aclose(self) -> None:
        """释放内部持有的 HTTP 客户端（若由外部注入则不处理）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

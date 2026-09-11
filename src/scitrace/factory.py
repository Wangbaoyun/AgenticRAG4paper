"""装配层：把配置变成一整套可用的对象。

## 为什么单独有这么一层

``pipeline`` 与 ``agent`` 只依赖 ``ports`` 中的协议，因此它们**不知道**具体用的是
numpy 向量库还是 tantivy、是本地嵌入还是远程端点。总得有人知道——那就是这里。

把这件"脏活"集中在一个模块，换来的是：换任何部件只改这里一处；
测试可以用 :class:`Services` 直接注入内存实现而不必碰配置文件；
以及最实际的一点——**依赖方向可以被机械检查**（见 ``tests/test_layering.py``）。

## 延迟构造

:func:`build_services` 只在需要时加载模型与索引：

- 嵌入模型（本地 sentence-transformers）首次加载要下载数百 MB，
  仅为了打印配置就触发下载是不可接受的；
- 三个角色的 LLM 客户端按需构造，未配置 API key 时**不报错**——
  报错应当发生在真正调用模型的那一刻，而不是在 ``stc config show`` 时。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from scitrace.adapters.indexes import NumpyVectorIndex, TantivyFullTextIndex
from scitrace.adapters.llm import LiteLLMClient, LocalEmbeddingClient, OpenAIEmbeddingClient
from scitrace.adapters.metadata import (
    CrossrefProvider,
    MetadataResolverImpl,
    OpenAlexProvider,
    RetractionProvider,
    SemanticScholarProvider,
)
from scitrace.config import LLMRole, Settings
from scitrace.domain import Source
from scitrace.domain.session import Usage
from scitrace.pipeline.retrieval import Retriever
from scitrace.pipeline.screening import CrossEncoderScreener, LLMScreener
from scitrace.pipeline.synthesis import AnswerSynthesizer
from scitrace.ports import (
    DocumentParser,
    EmbeddingClient,
    EvidenceScreener,
    LLMClient,
    MetadataProvider,
    MetadataResolver,
)
from scitrace.ports.reranker import Reranker
from scitrace.prompts import PromptSet, get_prompt_set
from scitrace.service import SourceStore

logger = logging.getLogger(__name__)

__all__ = ["Services", "build_metadata_providers", "build_services"]


@dataclass
class Services:
    """一整套装配好的对象。

    持有它意味着"一次问答所需的一切都已就绪"。释放责任也在这里：
    :meth:`aclose` 关闭所有持有外部资源的部件，顺序无关紧要但必须都关到——
    漏掉一个的后果是进程退出前挂着未关闭的连接或模型句柄。
    """

    settings: Settings
    prompts: PromptSet
    vector_index: NumpyVectorIndex
    fulltext_index: TantivyFullTextIndex
    embedder: EmbeddingClient
    llms: dict[LLMRole, LLMClient] = field(default_factory=dict)
    reranker: Reranker | None = None
    resolver: MetadataResolver | None = None
    retriever: Retriever | None = None
    screener: EvidenceScreener | None = None
    synthesizer: AnswerSynthesizer | None = None
    sources: dict[str, Source] = field(default_factory=dict)
    #: 一次问答的**累计**用量。筛选器与合成器各自只知道自己那部分，
    #: 由这里统一累加——成本可观测（增量 ④）要求汇总口径只有一处。
    usage: Usage = field(default_factory=Usage)

    @property
    def index_dir(self) -> Path:
        """当前配置对应的索引目录。"""
        return self.settings.index_dir

    def merge_usage(self, other: Usage) -> None:
        """把某个部件的用量并入会话累计。"""
        self.usage = self.usage.merge(other)

    def llm(self, role: LLMRole = "main") -> LLMClient:
        """取某个角色的 LLM 客户端。"""
        if role not in self.llms:
            raise KeyError(f"未装配角色 {role!r} 的 LLM 客户端")
        return self.llms[role]

    async def aclose(self) -> None:
        """释放全部资源。单个部件失败不影响其余。"""
        for name, closer in (
            ("元数据解析器", self.resolver.aclose if self.resolver else None),
            ("检索器", self.retriever.aclose if self.retriever else None),
            ("筛选器", self.screener.aclose if self.screener else None),
            ("合成器", self.synthesizer.aclose if self.synthesizer else None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception as error:  # noqa: BLE001 - 关闭失败不应阻断其余关闭
                logger.warning("关闭%s失败：%s", name, error)


def build_metadata_providers(settings: Settings) -> list[MetadataProvider]:
    """按配置构造元数据来源列表，顺序即优先级顺序。"""
    mailto = settings.metadata.crossref_mailto
    s2_key = (
        settings.metadata.semantic_scholar_api_key.get_secret_value()
        if settings.metadata.semantic_scholar_api_key
        else None
    )
    factories: dict[str, object] = {
        "crossref": lambda: CrossrefProvider(mailto=mailto),
        "semantic_scholar": lambda: SemanticScholarProvider(api_key=s2_key),
        "openalex": lambda: OpenAlexProvider(mailto=settings.metadata.openalex_mailto),
        "retraction": lambda: RetractionProvider(settings.metadata.retraction_csv),
    }

    providers: list[MetadataProvider] = []
    for name in settings.metadata.providers:
        factory = factories.get(name)
        if factory is None:
            # 配置里写错来源名必须报错：静默忽略会让"我配了撤稿检查"变成一句空话。
            raise ValueError(
                f"未知的元数据来源 {name!r}；可用来源：{', '.join(sorted(factories))}"
            )
        providers.append(factory())  # type: ignore[operator]
    if settings.metadata.retraction_csv is not None and "retraction" not in settings.metadata.providers:
        providers.append(RetractionProvider(settings.metadata.retraction_csv))
    return providers


def _build_embedder(settings: Settings) -> EmbeddingClient:
    if settings.embedding.backend == "local":
        return LocalEmbeddingClient(
            settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            query_prefix=settings.embedding.query_prefix,
            dimension=settings.embedding.dimension,
        )
    return OpenAIEmbeddingClient(
        settings.embedding.model,
        api_key=(
            settings.embedding.api_key.get_secret_value() if settings.embedding.api_key else None
        ),
        api_base=settings.embedding.api_base,
        batch_size=settings.embedding.batch_size,
        query_prefix=settings.embedding.query_prefix,
        dimension=settings.embedding.dimension,
    )


def _build_llms(settings: Settings) -> dict[LLMRole, LLMClient]:
    api_key = settings.llm.api_key.get_secret_value() if settings.llm.api_key else None
    clients: dict[LLMRole, LLMClient] = {}
    for role in ("main", "summary", "agent"):
        clients[role] = LiteLLMClient(
            settings.llm.model_for(role),  # type: ignore[arg-type]
            api_key=api_key,
            api_base=settings.llm.api_base,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout_s=settings.llm.timeout_s,
        )
    return clients


def build_services(
    settings: Settings,
    *,
    language: str = "zh",
    load_index: bool = True,
    with_reranker: bool | None = None,
) -> Services:
    """按配置装配一整套对象。

    Args:
        settings: 已校验的配置。
        language: 提示词语言。
        load_index: 是否从磁盘恢复既有索引。摄入路径应当设为
            ``False``（它要往空索引里写），查询路径保持 ``True``。
        with_reranker: 是否构造重排器；``None`` 表示按检索策略自动决定。
            刻意**不**在策略需要重排器时静默省略——那会让检索器在运行时报错，
            而错误发生在装配期更容易定位。

    Returns:
        装配好的 :class:`Services`。
    """
    prompts = get_prompt_set(language)
    index_dir = settings.index_dir

    vector_index = NumpyVectorIndex(path=index_dir)
    fulltext_index = TantivyFullTextIndex(index_dir / "fts")
    if load_index:
        vector_index.load()
        logger.debug("已恢复向量索引：%d 个片段", len(vector_index))

    embedder = _build_embedder(settings)
    llms = _build_llms(settings)

    need_reranker = (
        settings.retrieval.needs_reranker
        or settings.screening.backend == "cross_encoder"
    )
    reranker: Reranker | None = None
    if with_reranker is True or (with_reranker is None and need_reranker):
        from scitrace.adapters.llm.reranker import CrossEncoderReranker  # noqa: PLC0415

        reranker = CrossEncoderReranker(
            settings.retrieval.rerank_model
            if settings.retrieval.needs_reranker
            else settings.screening.cross_encoder_model
        )

    resolver: MetadataResolver | None = None
    if settings.metadata.enabled:
        resolver = MetadataResolverImpl(
            build_metadata_providers(settings),
            timeout_s=settings.metadata.timeout_s,
        )

    retriever = Retriever(
        vector_index=vector_index,
        fulltext_index=fulltext_index,
        embedder=embedder,
        settings=settings.retrieval,
        reranker=reranker,
    )

    if settings.screening.backend == "cross_encoder":
        if reranker is None:  # pragma: no cover - 由上面的构造逻辑保证
            raise ValueError("screening.backend=cross_encoder 需要重排器")
        screener: EvidenceScreener = CrossEncoderScreener(
            llm=llms["summary"],
            reranker=reranker,
            prompts=prompts,
            settings=settings.screening,
            max_evidence=settings.answer.max_evidence,
        )
    else:
        screener = LLMScreener(
            llm=llms["summary"],
            prompts=prompts,
            settings=settings.screening,
            max_evidence=settings.answer.max_evidence,
        )

    synthesizer = AnswerSynthesizer(
        llm=llms["main"], prompts=prompts, settings=settings.answer
    )

    sources = SourceStore(index_dir).load_sources() if load_index else {}

    return Services(
        settings=settings,
        prompts=prompts,
        vector_index=vector_index,
        fulltext_index=fulltext_index,
        embedder=embedder,
        llms=llms,
        reranker=reranker,
        resolver=resolver,
        retriever=retriever,
        screener=screener,
        synthesizer=synthesizer,
        sources=sources,
    )


def _resolve_parser(path: Path, *, preferred: str | None = None) -> DocumentParser:
    """装配层提供的解析器选择函数（唯一 import adapters.parsers 的地方）。"""
    from scitrace.adapters.parsers import select_parser  # noqa: PLC0415

    return select_parser(path, preferred=preferred)


def build_ingest_pipeline(settings: Settings, services: Services) -> object:
    """按配置装配摄入管线。

    单独一个函数而不是 :func:`build_services` 的一部分：摄入与查询用的是
    **同一批索引对象**，但摄入需要的是"往里写"的视角。
    """
    from scitrace.pipeline.ingest import IngestPipeline  # noqa: PLC0415

    return IngestPipeline(
        index_dir=settings.index_dir,
        chunking=settings.ingest.chunking,
        parser_resolver=partial(_resolve_parser, preferred=settings.ingest.parser),
        vector_index=services.vector_index,
        fulltext_index=services.fulltext_index,
        embedder=services.embedder,
        resolver=services.resolver,
        max_file_mb=settings.ingest.max_file_mb,
    )

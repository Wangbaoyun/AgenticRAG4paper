"""学术元数据来源的具体实现。

每个来源一个模块，各自封装一个公开 REST API。它们**只负责"我知道什么"**——
合并、优先级、降级策略都在 :mod:`scitrace.adapters.metadata.resolver` 里，
不在这里重复。

按"一个来源一个文件"划分，而不是把几个 API 塞进一个模块：每个来源的认证方式、
限流策略、字段嵌套形态、错误表现都不同，混在一起会让任何一处修改波及全部来源。
"""

from scitrace.adapters.metadata.crossref import CrossrefProvider
from scitrace.adapters.metadata.http_base import (
    HttpMetadataProvider,
    title_similarity,
)
from scitrace.adapters.metadata.openalex import OpenAlexProvider
from scitrace.adapters.metadata.resolver import (
    DEFAULT_FIELD_PRIORITY,
    MetadataResolverImpl,
)
from scitrace.adapters.metadata.retraction import RetractionProvider
from scitrace.adapters.metadata.semantic_scholar import SemanticScholarProvider

__all__ = [
    "DEFAULT_FIELD_PRIORITY",
    "CrossrefProvider",
    "HttpMetadataProvider",
    "MetadataResolverImpl",
    "OpenAlexProvider",
    "RetractionProvider",
    "SemanticScholarProvider",
    "title_similarity",
]

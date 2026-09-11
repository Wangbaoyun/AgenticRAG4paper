"""流水线：把领域模型与端口组装成可执行的业务步骤。

本包的各模块**只依赖 ``domain``、``ports`` 与 ``util``**，不直接依赖任何适配器实现。
这样"换一个 PDF 解析后端""换一个向量库""换一个筛选策略"都不会波及这里的任何一行——
依赖方向由装配层（``scitrace.factory``）负责在运行时注入具体实现。

这条约束是可测试的（``tests/test_layering.py``），因为"pipeline 不 import adapters"
一旦被破坏，替换实现的能力就会在不知不觉中消失。
"""

from scitrace.pipeline.chunking import chunk_document, split_sentences
from scitrace.pipeline.ingest import IngestPipeline, IngestReport, Manifest, ManifestStore
from scitrace.pipeline.retrieval import Retriever, reciprocal_rank_fusion
from scitrace.pipeline.screening import CrossEncoderScreener, LLMScreener, coerce_relevance

__all__ = [
    "CrossEncoderScreener",
    "IngestPipeline",
    "IngestReport",
    "LLMScreener",
    "Manifest",
    "ManifestStore",
    "Retriever",
    "chunk_document",
    "coerce_relevance",
    "reciprocal_rank_fusion",
    "split_sentences",
]

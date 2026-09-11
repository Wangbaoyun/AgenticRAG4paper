"""流水线：把领域模型与端口组装成可执行的业务步骤。

本包的各模块**只依赖 ``domain`` 与 ``ports``**，不直接依赖任何适配器实现。
这样"换一个 PDF 解析后端"或"换一个向量库"不会波及这里的任何一行——
依赖方向由装配层（``scitrace.factory``）负责在运行时注入具体实现。
"""

from scitrace.pipeline.chunking import chunk_document, split_sentences

__all__ = ["chunk_document", "split_sentences"]

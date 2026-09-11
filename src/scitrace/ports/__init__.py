"""接口协议层：本项目所有"可替换部件"的契约。

设计原则（对应 SPEC §1.2 的"可插拔"要求）：

- **协议在 ports，实现在 adapters**。pipeline 与 agent 只依赖 ports 中的
  ``typing.Protocol``，因此"换 PDF 解析器""换向量库""换重排后端"都不触及业务逻辑。
- **Protocol 而非 ABC**：实现方不需要 import 本包，也不需要继承任何基类。
  这让第三方（或本项目的另一个实现）可以在完全不了解 scitrace 的前提下满足契约。
- **端口只用领域类型**：签名中出现的一律是 ``domain`` 中的模型或 stdlib 容器，
  不出现 numpy 数组、HTTP 响应、数据库游标等实现细节。
"""

from scitrace.ports.common import ScoredFragment, ToolSpec
from scitrace.ports.fulltext_index import FullTextIndex
from scitrace.ports.llm import EmbeddingClient, LLMClient, LLMMessage, LLMResponse, ToolCall
from scitrace.ports.metadata import Confidence, MetadataMatch, MetadataProvider, MetadataResolver
from scitrace.ports.parser import DocumentParser, ParseError
from scitrace.ports.screener import EvidenceScreener
from scitrace.ports.vector_index import VectorIndex

__all__ = [
    "Confidence",
    "DocumentParser",
    "EmbeddingClient",
    "EvidenceScreener",
    "FullTextIndex",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "MetadataMatch",
    "MetadataProvider",
    "MetadataResolver",
    "ParseError",
    "ScoredFragment",
    "ToolCall",
    "ToolSpec",
    "VectorIndex",
]

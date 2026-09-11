"""LLM 与嵌入客户端适配器。

本包把 :mod:`scitrace.ports.llm` 的两个协议落到具体后端上。实现上刻意分成三层：

1. **纯转换**（:mod:`scitrace.adapters.llm.convert`）——消息格式、工具调用解析、
   向量归一化。没有任何 I/O，因此可以被穷尽测试，包括各种畸形输入；
2. **重试策略**（:mod:`scitrace.adapters.llm.retry`）——哪些错误值得重试、
   退避多久。同样纯逻辑；
3. **网络客户端**（:mod:`scitrace.adapters.llm.litellm_client` 等）——只负责发请求。

这样划分的收益在测试上最明显：LLM 相关代码最容易出错的部分（模型输出的 JSON 参数
不合法、用量字段缺失、向量忘了归一化）全部落在第 1 层，不需要联网就能覆盖。
"""

from scitrace.adapters.llm.convert import (
    normalize_vector,
    parse_tool_calls,
    to_backend_messages,
)
from scitrace.adapters.llm.embedding import LocalEmbeddingClient, OpenAIEmbeddingClient
from scitrace.adapters.llm.litellm_client import LiteLLMClient
from scitrace.adapters.llm.retry import RetryPolicy, is_retryable_error

__all__ = [
    "LiteLLMClient",
    "LocalEmbeddingClient",
    "OpenAIEmbeddingClient",
    "RetryPolicy",
    "is_retryable_error",
    "normalize_vector",
    "parse_tool_calls",
    "to_backend_messages",
]

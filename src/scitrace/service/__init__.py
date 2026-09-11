"""本地持久化：索引之外的"小数据"。

索引本身（向量、全文）由各自的适配器负责落盘；本包负责的是围绕它们的**元数据**：

- :class:`SourceStore` —— 文献书目信息。查询阶段要靠它把 ``source_key``
  渲染成可读的文内引用与 BibTeX，所以它必须与索引一同持久化。
- :class:`SessionStore` —— 问答会话历史（M6 使用）。

放在 ``service`` 而不是 ``pipeline``：摄入与查询**都要**读写它们，
放在任一侧都会让另一侧产生反向依赖。
"""

from scitrace.service.store import SessionStore, SourceStore

__all__ = ["SessionStore", "SourceStore"]

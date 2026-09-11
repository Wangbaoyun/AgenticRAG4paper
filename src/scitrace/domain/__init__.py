"""领域模型：纯数据结构，零外部依赖（pydantic 除外），不含任何 I/O。

术语定义见 docs/SPEC.md §2.1。本包的模型是全部 pipeline / agent / adapter 的公共语言：
适配器把外部世界翻译成本包的类型，pipeline 只在本包的类型上运算。
"""

from scitrace.domain.citation import Citation, render_bibtex, render_inline_citation
from scitrace.domain.evidence import Evidence, make_evidence_key
from scitrace.domain.fragment import Fragment, ParsedDocument, ParsedPage
from scitrace.domain.session import (
    Answer,
    Session,
    SessionStatus,
    StageTiming,
    Usage,
)
from scitrace.domain.source import Source, SourceKey, SourcePatch, make_source_key

__all__ = [
    "Answer",
    "Citation",
    "Evidence",
    "Fragment",
    "ParsedDocument",
    "ParsedPage",
    "Session",
    "SessionStatus",
    "Source",
    "SourceKey",
    "SourcePatch",
    "StageTiming",
    "Usage",
    "make_evidence_key",
    "make_source_key",
    "render_bibtex",
    "render_inline_citation",
]

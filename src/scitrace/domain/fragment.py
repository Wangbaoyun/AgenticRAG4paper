"""解析产物与文本片段。

**刻意的设计决定：``Fragment`` 不携带向量。** 向量存放在向量索引内部
（``adapters/indexes``），理由有三：

1. 领域模型保持对 numpy 的零依赖，domain 层可以用纯 stdlib 思维审阅与测试；
2. 每个片段 768–1024 维的浮点数组若进 JSON 序列化，`fragments.jsonl` 会膨胀数百倍，
   而索引重建时真正需要持久化的是**文本**，向量可以重算或被缓存单独管理；
3. 向量维度取决于 embedding 模型，把它放进领域模型会让"换模型"变成领域模型变更。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scitrace.domain.source import SourceKey
from scitrace.util.hashing import derive_key, normalize_text

__all__ = ["Fragment", "ParsedDocument", "ParsedPage", "make_fragment_id"]


def make_fragment_id(*, source_key: SourceKey, section_path: list[str], chunk_index: int) -> str:
    """派生片段的确定性 id。

    组成：``source_key`` + 章节路径 + 块序号。**不含文本内容**——这样
    "同一位置被重新切分"会得到不同的 ``chunk_index`` 而自然区分，
    同时"重新解析同一篇论文得到相同分块"会得到相同 id，使增量索引可以按 id 去重。
    """
    return derive_key(source_key, "/".join(section_path), str(chunk_index))


class ParsedPage(BaseModel):
    """解析出的单页文本。

    ``page_number`` 为 **1-based**，与人的阅读习惯一致；引用渲染时直接使用，
    不需要 ±1 转换——这类偏移错误是引用不可信的主要来源之一。
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class ParsedDocument(BaseModel):
    """解析器（``ports.parser.DocumentParser``）的统一输出。

    解析器**只负责把文件变成文本**，不做分块、不做元数据补全、不碰索引——
    这三件事分别属于 ``pipeline.chunking`` 与 ``adapters.metadata``。
    保持解析器的单一职责，才能让"换 PDF 解析后端"不影响其余任何环节。
    """

    model_config = ConfigDict(extra="forbid")

    pages: list[ParsedPage]
    hints: dict[str, str] = Field(
        default_factory=dict,
        description="从文件内嵌元数据/文件名提取的候选书目信息，供元数据补全作为起点",
    )
    parser: str = Field(description="产生本结果的解析器名，用于指纹与排障")

    @property
    def n_pages(self) -> int:
        """页数。"""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """全文（按页拼接）。供需要全文视图的上层使用（如参考文献区块剔除）。"""
        return "\n\n".join(page.text for page in self.pages)


class Fragment(BaseModel):
    """文献切分出的一个文本块——检索与引用的最小单位。"""

    model_config = ConfigDict(extra="forbid")

    fragment_id: str
    source_key: SourceKey
    text: str
    chunk_index: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    char_count: int = Field(ge=0)
    media: list[dict[str, object]] = Field(
        default_factory=list,
        description="v1 预留：多模态富化产物（图片/表格描述）。首版恒为空列表",
    )

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        cleaned = normalize_text(value)
        if not cleaned:
            raise ValueError("Fragment.text 不能为空")
        return cleaned

    @property
    def page_label(self) -> str:
        """把页码范围渲染为引用中的位置短语。

        Returns:
            ``"pages 3-4"`` / ``"page 3"``；无页码信息时返回空串，
            由引用渲染层决定省略位置部分。
        """
        if self.page_start is None:
            return ""
        if self.page_end is None or self.page_end == self.page_start:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"

    @property
    def section_label(self) -> str:
        """章节路径的可读形式，如 ``"2 Method > 2.1 Retrieval"``。"""
        return " > ".join(self.section_path)

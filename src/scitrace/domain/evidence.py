"""证据：经过相关性筛选的片段 + 摘要 + 评分。

证据是"检索"与"合成"之间的契约对象。它比 :class:`~scitrace.domain.fragment.Fragment`
多出三样东西，每一样都直接服务于"答案可溯源"这一核心目标：

1. ``key``：**确定性**的引用键。相同片段在任何运行中产生相同键，因此
   "上一轮答案里的引用"在下一轮仍然可解释，会话记录也可被机器校验。
2. ``summary``：LLM 在**看过问题之后**对片段做的定向摘要。它是送给合成模型的
   实际内容——原文片段往往数千字，直接塞进上下文既昂贵又稀释信号。
3. ``relevance``：0–10 的相关性评分，是过滤与排序的依据（SPEC §3.7）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scitrace.domain.source import SourceKey
from scitrace.util.hashing import sha256_hex

__all__ = ["EVIDENCE_KEY_PREFIX", "Evidence", "make_evidence_key"]

#: 引用键前缀。刻意与任何上游实现的引用方案不同（SPEC §10），
#: 既是可读性设计，也让审计脚本能机械地识别命名污染。
EVIDENCE_KEY_PREFIX = "ev-"

#: 相关性评分的取值范围。
MIN_RELEVANCE = 0
MAX_RELEVANCE = 10

#: 评分为此值时表示"筛选阶段判定为不相关"。
IRRELEVANT = 0


def make_evidence_key(source_key: SourceKey, fragment_id: str) -> str:
    """从 (文献, 片段) 派生确定性的引用键，形如 ``ev-1a2b3c4d``。

    键的稳定性带来两个好处：

    - 审计引用：给定一段答案，可以机械校验每个引用键是否对应真实存在的证据；
    - 幂等去重：同一片段被多轮检索召回时，键相同，天然去重而不需要额外状态。
    """
    return f"{EVIDENCE_KEY_PREFIX}{sha256_hex(f'{source_key}:{fragment_id}', length=8)}"


class Evidence(BaseModel):
    """一条可用于支撑答案结论的证据。"""

    model_config = ConfigDict(extra="forbid")

    key: str
    source_key: SourceKey
    fragment_id: str
    summary: str
    relevance: int = Field(ge=MIN_RELEVANCE, le=MAX_RELEVANCE)
    citation: str = Field(
        default="",
        description="文内引用文本（如 '(author2024title pages 3-4)'），由引用渲染层填充",
    )
    page_label: str = Field(default="", description="位置短语，随片段一同带入以免二次查询")
    section_path: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _clean_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Evidence.summary 不能为空")
        return cleaned

    @property
    def is_relevant(self) -> bool:
        """是否被筛选阶段判定为相关（评分 > 0）。

        注意：**是否入选上下文**由配置阈值 ``screening.min_relevance`` 决定，
        而不是由本属性决定——把策略留在配置层，本属性只表达"这条证据有没有信号"。
        """
        return self.relevance > IRRELEVANT

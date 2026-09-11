"""证据筛选契约 —— 可插拔检索内核的关键抽象点（SPEC §9 增量 ②）。

"召回之后如何挑出真正有用的片段"是整个系统性价比最敏感的环节：
它决定了送进合成模型的上下文质量，也决定了每次问答的 token 开销。
把这一步收敛成单一协议，使三种做法（LLM 打分摘要 / Cross-Encoder 重排 /
两者组合）可以互换并做消融对照，而不必改动任何其他模块。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from scitrace.domain import Evidence, Fragment, Source, SourceKey

__all__ = ["EvidenceScreener"]


@runtime_checkable
class EvidenceScreener(Protocol):
    """把候选片段筛选为带摘要与评分的证据。"""

    @property
    def name(self) -> str:
        """实现标识（如 ``"llm"`` / ``"cross_encoder"``），用于实验记录与消融表。"""
        ...

    async def screen(
        self,
        question: str,
        fragments: Sequence[Fragment],
        *,
        sources: Mapping[SourceKey, Source],
    ) -> list[Evidence]:
        """筛选并加工候选片段。

        Args:
            question: 用户问题。**必须**参与筛选——"这段文本是否与问题相关"
                离开问题就没有定义。同一片段对不同问题会得到不同评分。
            fragments: 召回得到的候选片段。
            sources: ``source_key -> Source``，用于填充
                ``Evidence.citation`` 与页码等渲染信息。传映射而不是让
                筛选器自己去查库，是为了让本协议保持无 I/O 之外的依赖。

        Returns:
            证据列表，**已按相关性降序排列**并已应用
            ``screening.min_relevance`` 阈值。返回空列表是完全正常的结果
            （意味着"没有找到相关证据"），调用方据此走向拒答路径。

        Note:
            实现必须容忍**部分失败**：单个片段的 LLM 输出无法解析时，
            应把该片段丢弃并计入 ``Usage.parse_failures``，
            而不是让整批失败（SPEC §3.7 容错链、§3.10 边界行为）。
        """
        ...

    async def aclose(self) -> None:
        """释放资源（连接池、模型句柄等）。"""
        ...

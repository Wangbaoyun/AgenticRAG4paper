"""学术元数据来源契约。

每个 provider 只回答一个问题："关于这篇文献，你知道什么？"
它**不做**合并、不做优先级判断、不写数据库——合并策略属于
``pipeline``/``adapters.metadata`` 的组合层（SPEC §3.5）。

这个划分的意义在于：新增一个数据源（比如中文的 CNKI、或 bioRxiv）
只需实现本协议并注册，不触碰合并与降级逻辑。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from scitrace.domain import SourcePatch

__all__ = ["MetadataProvider"]


@runtime_checkable
class MetadataProvider(Protocol):
    """单一元数据来源。"""

    @property
    def name(self) -> str:
        """来源标识（如 ``"crossref"``）。

        会写入 ``Source.metadata_sources``，用于回答"这条元数据是谁提供的"
        ——当一条错误标题出现时，能否快速定位到来源直接决定排查效率。
        """
        ...

    async def lookup(self, patch: SourcePatch) -> SourcePatch | None:
        """按已知线索查询元数据。

        查询线索来自"当前已知的稀疏补丁"（通常含 ``doi`` 与/或 ``title``）。
        实现应根据自己能用的线索选择端点：有 DOI 时走精确端点，
        只有标题时走模糊检索端点。

        Args:
            patch: 已知线索。实现**不应**修改它。

        Returns:
            找到的稀疏补丁；未找到或无可用线索时返回 ``None``。

        Raises:
            实现**不应**因网络错误、限流或响应格式异常而抛出——
            SPEC §3.5 要求"任一 provider 失败不影响主流程"。
            实现方应自行捕获、记录 warning 并返回 ``None``。
            把降级责任放在实现内部，是为了让组合层不必为每个 provider
            写一遍 try/except（那必然会漏掉某一个）。
        """
        ...

    async def aclose(self) -> None:
        """释放连接等资源。未持有资源时可为空实现。"""
        ...

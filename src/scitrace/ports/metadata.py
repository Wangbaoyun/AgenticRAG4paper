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

__all__ = ["MetadataProvider", "MetadataResolver"]


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


@runtime_checkable
class MetadataResolver(Protocol):
    """多个 :class:`MetadataProvider` 的组合与合并。

    与 provider 的分工：provider 只回答"我知道什么"，resolver 负责
    **按优先级合并、处理降级、并保证不覆盖已有非空字段**（SPEC §3.5）。

    单独抽出这一层，是因为合并策略是有状态、有顺序、需要单测的逻辑，
    把它塞进每个 provider 会让 N 个数据源产生 N 份略有差异的合并实现。
    """

    @property
    def name(self) -> str:
        """组合器标识，用于日志与 ``Source.metadata_sources`` 的来源说明。"""
        ...

    async def enrich(self, patch: SourcePatch) -> SourcePatch:
        """用全部已注册的 provider 补全给定补丁。

        Args:
            patch: 已知线索（通常含 DOI、标题、年份）。实现**不应**修改它。

        Returns:
            合并后的补丁。**并发**执行各 provider，按配置的优先级顺序依次
            ``fill_gaps_from``；任一 provider 失败或超时都只降级为"该来源无贡献"，
            不影响其他来源，也不抛出异常（SPEC §3.5 降级要求）。
        """
        ...

    async def aclose(self) -> None:
        """释放全部 provider 持有的资源。"""
        ...

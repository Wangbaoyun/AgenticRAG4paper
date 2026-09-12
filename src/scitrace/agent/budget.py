"""预算闸门：会话级的 token 与成本上限。【原创增量 ④】

## 闸门与统计的区别

事后统计只能告诉你花了多少钱，拦不住继续花。闸门要在**下一次模型调用之前**
做出判断，并且超限时的动作不是抛异常——而是**用已有证据强制收尾**。

这个取舍很重要：一次问答已经消耗了几十万 token 与几分钟，
因为超了 5% 的预算就把全部工作丢弃、返回一个错误，是把用户已经付出的成本
变成沉没成本。正确做法是产出一个基于已有证据的答案（状态标为 ``TRUNCATED``），
让用户拿到"不完整但可用"的结果，并明确知道它被截断了。
"""

from __future__ import annotations

import math

from scitrace.domain.session import Usage

__all__ = ["Budget"]

#: 浮点比较容差。恰好等于上限**不算**超限——"用到上限"是正常结束，
#: 而浮点累加会让 0.1+0.2 这类运算产生 1e-17 量级的误差，
#: 不设容差就会出现"明明刚好用完却报超限"的诡异行为。
_EPSILON = 1e-9


class Budget:
    """会话级预算。两个上限都可以为 ``None``，表示不限制。"""

    def __init__(self, *, max_tokens: int | None = None, max_cost_usd: float | None = None) -> None:
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens 必须为正数或 None，得到 {max_tokens}")
        if max_cost_usd is not None and max_cost_usd < 0:
            raise ValueError(f"max_cost_usd 必须为非负数或 None，得到 {max_cost_usd}")
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd

    @property
    def unlimited(self) -> bool:
        """是否完全没有限制。"""
        return self.max_tokens is None and self.max_cost_usd is None

    def check(self, usage: Usage) -> str | None:
        """检查是否超限。

        Args:
            usage: 到目前为止的累计用量。

        Returns:
            超限原因（写入 ``Session.notes``）；未超限返回 ``None``。
        """
        if self.max_tokens is not None and usage.total_tokens > self.max_tokens + _EPSILON:
            return "token 预算超限"
        if not usage.cost_known:
            # 成本不可信时**不用它做判断**：拿一个恒为 0 的数字去比上限，
            # 结果永远是不超限——那是一个假装在工作的闸门。
            # 调用方（AgentRuntime）会就此告警一次，把"成本治理在当前配置下失效"
            # 这件事明确告诉用户，而不是让它悄悄过去。
            return None
        if (
            self.max_cost_usd is not None
            and not math.isclose(usage.estimated_cost_usd, self.max_cost_usd, abs_tol=_EPSILON)
            and usage.estimated_cost_usd > self.max_cost_usd
        ):
            return "成本预算超限"
        return None

    @property
    def cost_gate_active(self) -> bool:
        """成本上限是否被配置了。"""
        return self.max_cost_usd is not None

    def exhausted(self, usage: Usage) -> bool:
        """是否已经超限。"""
        return self.check(usage) is not None

    def __repr__(self) -> str:  # pragma: no cover - 仅用于排障
        return f"Budget(max_tokens={self.max_tokens}, max_cost_usd={self.max_cost_usd})"

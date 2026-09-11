"""重试策略：区分"值得重试"与"重试也没用"的错误。

朴素实现（"失败就重试 N 次"）在 LLM 场景下有两个具体危害：

- **重试不可恢复的错误**：API key 无效、上下文超长、内容被安全策略拦截——
  这些错误重试一百次结果都一样，却会把一次本应立即失败的操作拖成几分钟的等待，
  期间用户完全不知道发生了什么；
- **重试不带抖动**：多个并发请求同时失败后按固定间隔重试，会形成同步的请求波峰，
  把限流从"偶发"放大成"持续"。
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

__all__ = ["RetryPolicy", "is_retryable_error", "with_retries"]

T = TypeVar("T")

#: 值得重试的 HTTP 状态码。408/409/425 是暂时性冲突，429 是限流，
#: 5xx 与 529（部分提供商的"过载"）是服务端问题。
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: 明确**不可**重试的错误标记。命中即立刻失败，不做无谓等待。
NON_RETRYABLE_MARKERS: tuple[str, ...] = (
    "invalid_api_key",
    "invalid api key",
    "authenticationerror",
    "authentication_error",
    "permissiondenied",
    "insufficient_quota",
    "context_length_exceeded",
    "contextlength",
    "content_filter",
    "contentfilter",
    "invalid_request_error",
    "badrequest",
    "notfounderror",
    "model_not_found",
    "unsupported",
)

#: 值得重试的错误标记（当拿不到状态码时按消息判断）。
RETRYABLE_MARKERS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "rate limit",
    "ratelimit",
    "overloaded",
    "temporarily unavailable",
    "server error",
    "service unavailable",
    "bad gateway",
    "try again",
)


@dataclass(frozen=True)
class RetryPolicy:
    """指数退避 + 抖动的重试参数。"""

    attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    jitter: float = 0.25

    def delay_for(self, attempt: int) -> float:
        """返回第 ``attempt`` 次失败后的等待秒数（``attempt`` 从 1 开始）。

        指数退避：``base * 2^(attempt-1)``，上限 ``max_delay_s``，
        再乘以 ``[1-jitter, 1+jitter]`` 的随机因子。
        """
        raw = min(self.base_delay_s * (2 ** max(0, attempt - 1)), self.max_delay_s)
        if self.jitter <= 0:
            return raw
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))  # noqa: S311 - 抖动无需密码学随机


def is_retryable_error(error: BaseException) -> bool:
    """判断一个异常是否值得重试。

    判定顺序（先精确后模糊）：

    1. 异常上带 ``status_code`` / ``status`` → 按状态码判断；
    2. 消息命中"不可重试"标记 → ``False``（**优先于**可重试标记，
       因为 ``invalid_request_error`` 这类错误的消息里常同时出现
       "rate limit" 之类的字样）；
    3. 异常类型名或消息命中"可重试"标记 → ``True``；
    4. 其余默认 ``False``：不确定时不重试，避免把偶发错误变成长时间挂起。

    Args:
        error: 捕获到的异常。

    Returns:
        是否应当重试。
    """
    for attribute in ("status_code", "status", "http_status"):
        status = getattr(error, attribute, None)
        if isinstance(status, int):
            return status in RETRYABLE_STATUS_CODES

    text = f"{type(error).__name__} {error}".lower()
    if any(marker in text for marker in NON_RETRYABLE_MARKERS):
        return False
    return any(marker in text for marker in RETRYABLE_MARKERS)


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    description: str = "LLM 调用",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """执行 ``operation``，按 ``policy`` 重试可恢复错误。

    Args:
        operation: 无参异步可调用对象。**每次重试都会重新调用它**，
            因此它必须是无副作用的（或副作用可安全重放）。
        policy: 重试参数。
        description: 用于日志的说明。
        sleep: 睡眠函数，注入以便测试（测试中传入一个只记录不等待的实现）。

    Returns:
        ``operation`` 的返回值。

    Raises:
        BaseException: 最后一次尝试的异常原样抛出。**不包装成自定义异常**——
            上层需要按原始错误类型做降级判断（例如"该 provider 失败就跳过"），
            包装会抹掉这个信息。
    """
    last_error: BaseException | None = None
    for attempt in range(1, max(1, policy.attempts) + 1):
        try:
            return await operation()
        except BaseException as error:  # noqa: BLE001 - 需要按策略重新抛出
            last_error = error
            if attempt >= policy.attempts or not is_retryable_error(error):
                raise
            delay = policy.delay_for(attempt)
            logger.warning(
                "%s 第 %d/%d 次失败（%s），%.1f 秒后重试",
                description,
                attempt,
                policy.attempts,
                error,
                delay,
            )
            await sleep(delay)

    # 理论上不可达（循环内必然 return 或 raise），仅为类型收窄
    assert last_error is not None  # noqa: S101
    raise last_error

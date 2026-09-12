"""基于 LiteLLM 的 LLM 客户端。

LiteLLM 负责"把统一的调用翻译成一百多个提供商的 API"，本类负责**它不该管的事**：

- 把领域消息与工具声明翻译过去，把返回结果翻译回来（含畸形输出的容错）；
- 按 :mod:`scitrace.adapters.llm.retry` 的策略重试；
- **把用量与成本记全**（SPEC §9 增量 ④）。成本可观测的前提是每一次调用都被计量，
  所以这里连"用量字段缺失"的情况也要处理成"记 0 并告警"，而不是让字段变成 ``None``
  在统计时炸掉。
"""

from __future__ import annotations

import logging
from typing import Any

from scitrace.adapters.llm.convert import parse_tool_calls, strip_reasoning_tags, to_backend_messages
from scitrace.adapters.llm.retry import RetryPolicy, with_retries
from scitrace.config.settings import PricingSettings
from scitrace.ports import LLMMessage, LLMResponse, ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["LiteLLMClient"]

#: 已经就"无法估算成本"告警过的模型名。
#:
#: 真实运行中这条告警会**每次调用都触发**（一次问答十几次），把终端刷满，
#: 反而掩盖了真正需要注意的日志。成本拿不到是配置层面的既定事实，
#: 报一次即可。
_COST_WARNING_ISSUED: set[str] = set()


class LiteLLMClient:
    """满足 :class:`~scitrace.ports.LLMClient` 协议的 LiteLLM 后端。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        retry_policy: RetryPolicy | None = None,
        pricing: PricingSettings | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy or RetryPolicy()
        #: 自备计价表。litellm 的价格表不收录自建/代理模型名，
        #: 缺失时它只能报告"无法估算"，而成本闸门会因此永远不触发。
        self._pricing = pricing or PricingSettings()

    @property
    def model_name(self) -> str:
        """模型标识。会进入用量报告，也用于判断配置是否被真正生效。"""
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_object: bool = False,
    ) -> LLMResponse:
        """调用模型并返回规范化结果。

        Raises:
            ImportError: 未安装 ``litellm``（可选依赖）。
            Exception: 网络或提供商错误，按重试策略重试后仍失败则原样抛出——
                上层据此决定降级（筛选阶段跳过该片段 / 合成阶段标记 FAIL）。
        """
        try:
            import litellm  # noqa: PLC0415 - 可选依赖，延迟导入
        except ImportError as error:  # pragma: no cover - 依赖缺失路径
            raise ImportError(
                "未安装 litellm。请执行 `pip install scitrace[llm]`，"
                "或改用 `openai` 后端。"
            ) from error

        request: dict[str, Any] = {
            "model": self._model,
            "messages": to_backend_messages(messages),
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "timeout": self._timeout_s,
        }
        if self._api_key:
            request["api_key"] = self._api_key
        if self._api_base:
            request["api_base"] = self._api_base
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools
            ]
        if json_object:
            request["response_format"] = {"type": "json_object"}

        async def call() -> Any:
            return await litellm.acompletion(**request)

        raw = await with_retries(
            call, policy=self._retry_policy, description=f"LLM 调用（{self._model}）"
        )
        return self._to_response(raw, litellm)

    def _to_response(self, raw: Any, litellm_module: Any) -> LLMResponse:
        """把 LiteLLM 的响应翻译成领域模型。

        对每一个字段都做"取不到就用默认值"的处理。不同提供商的响应结构差异很大
        （有的没有 ``usage``、有的 ``content`` 为 ``None``），
        在这里统一吸收掉，上层的成本统计与答案解析才能只写一遍。
        """
        choices = getattr(raw, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None

        content = getattr(message, "content", "") or ""
        tool_calls = parse_tool_calls(getattr(message, "tool_calls", None))
        finish_reason = str(getattr(choices[0], "finish_reason", "") or "") if choices else ""

        usage = getattr(raw, "usage", None)
        prompt_tokens = _as_int(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _as_int(getattr(usage, "completion_tokens", 0))
        cached_tokens = _as_int(
            getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
        )
        cached_tokens = min(cached_tokens, prompt_tokens)

        cost, currency, cost_known = _estimate_cost(
            raw, litellm_module, pricing=self._pricing, cached_tokens=cached_tokens,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )
        return LLMResponse(
            content=strip_reasoning_tags(content),
            tool_calls=tool_calls,
            model=str(getattr(raw, "model", "") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            cost_currency=currency,
            cost_known=cost_known,
            finish_reason=finish_reason,
        )


def _as_int(value: Any) -> int:
    """把用量字段安全地转成非负整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _estimate_cost(
    raw: Any,
    litellm_module: Any,
    *,
    pricing: PricingSettings,
    cached_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> tuple[float, str, bool]:
    """估算本次调用的成本，并**如实报告成本是否可信**。

    两级来源，按可信度排序：

    1. **自备计价表**（``settings.pricing``）。优先使用它——用户显式配置的价目
       比 litellm 内置的更贴近实际账单（尤其是自建/代理模型）。
    2. **litellm 内置价格表**。仅在未配置自备表时使用。

    两者都没有时返回 ``cost_known=False``。这一点比"记 0"重要得多：
    只记 0 并告警的话，``Usage.estimated_cost`` 恒为 0，上层的成本闸门
    会**看起来在工作、实际永远不触发**——本项目的实机运行正是如此
    （``deepseek-v4-flash`` 不在 litellm 表中，一次 agentic 问答烧掉 171k token
    而成本闸门全程沉默）。

    Returns:
        ``(成本, 币种, 是否可信)``。
    """
    if pricing.configured:
        uncached = max(0, prompt_tokens - cached_tokens)
        return (
            pricing.cost_of(
                cached_tokens=cached_tokens,
                uncached_tokens=uncached,
                completion_tokens=completion_tokens,
            ),
            pricing.currency,
            True,
        )

    try:
        cost = litellm_module.completion_cost(completion_response=raw)
    except Exception as error:  # noqa: BLE001 - 价格表缺项是常见情况
        model = str(getattr(raw, "model", "") or "unknown")
        if model not in _COST_WARNING_ISSUED:
            _COST_WARNING_ISSUED.add(model)
            logger.warning(
                "无法估算成本（模型 %s 既不在 litellm 价格表中、也未配置 pricing）：%s。"
                "本次及后续同类调用将记为 0，成本闸门不会生效。"
                "如需成本治理，请配置 settings.pricing。",
                model,
                error,
            )
        return 0.0, pricing.currency, False
    try:
        return max(0.0, float(cost)), "USD", True
    except (TypeError, ValueError):
        return 0.0, pricing.currency, False


def _as_int(value: Any) -> int:
    """把用量字段安全地转成非负整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)

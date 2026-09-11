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
from scitrace.ports import LLMMessage, LLMResponse, ToolSpec

logger = logging.getLogger(__name__)

__all__ = ["LiteLLMClient"]


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
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._retry_policy = retry_policy or RetryPolicy()

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

    @staticmethod
    def _to_response(raw: Any, litellm_module: Any) -> LLMResponse:
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

        return LLMResponse(
            content=strip_reasoning_tags(content),
            tool_calls=tool_calls,
            model=str(getattr(raw, "model", "") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(raw, litellm_module),
            finish_reason=finish_reason,
        )


def _as_int(value: Any) -> int:
    """把用量字段安全地转成非负整数。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _estimate_cost(raw: Any, litellm_module: Any) -> float:
    """估算本次调用的美元成本。

    成本拿不到时返回 ``0.0`` 并告警，**不抛异常**：成本是观测指标，
    不该因为某个提供商不在 LiteLLM 的价格表里就让整次问答失败。
    代价是"未知模型"会被统计成免费——因此用 warning 让它可见。
    """
    try:
        cost = litellm_module.completion_cost(completion_response=raw)
    except Exception as error:  # noqa: BLE001 - 价格表缺项是常见情况
        logger.warning("无法估算调用成本（该模型可能不在价格表中）：%s", error)
        return 0.0
    try:
        return max(0.0, float(cost))
    except (TypeError, ValueError):
        return 0.0

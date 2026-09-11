"""纯转换逻辑：消息编解码、工具调用解析、向量归一化。

本模块**不含任何 I/O**，因此可以被穷尽测试。把这一层单独拆出来是有明确动机的：
LLM 集成里真正容易出错的地方几乎全在这里，而它们恰恰是最难在联网环境下复现的
（模型的畸形输出不会稳定重现）。做成纯函数后，畸形输入变成了普通的参数化测试用例。
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from scitrace.ports import LLMMessage, ToolCall
from scitrace.util.text import (
    extract_balanced_json_object,
    parse_json_object,
    strip_reasoning_tags,
)

logger = logging.getLogger(__name__)

__all__ = [
    # 从 util.text 转出，供历史调用方与测试继续从本模块导入
    "extract_balanced_json_object",
    "normalize_vector",
    "parse_json_object",
    "parse_tool_arguments",
    "parse_tool_calls",
    "strip_reasoning_tags",
    "to_backend_messages",
]

def parse_tool_arguments(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """解析工具调用的参数串。

    模型给出的参数不是合法 JSON 是常态而非例外（尾逗号、单引号、缺引号的键、
    中文引号、被截断）。这里按"从严格到宽松"的顺序尝试，全部失败时返回原始串，
    由 Agent 循环决定是重试还是降级——**不在这里抛异常**，
    因为一次参数解析失败不该中断整个会话。

    Args:
        raw: 模型给出的参数串。

    Returns:
        ``(解析结果, 失败时的原始串)``。成功时第二项为 ``None``；
        失败时第一项为 ``None``、第二项为原始文本。
    """
    if raw is None or not raw.strip():
        return {}, None

    parsed = parse_json_object(raw)
    if parsed is not None:
        return parsed, None

    # 顶层是数组的情况：包一层，让上层拿到确定的结构而不是拿到 list 再各自判断。
    # 注意 extract_balanced_json_object 只识别花括号，数组要单独走一遍。
    for candidate in _plain_candidates(raw):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, list):
            return {"value": loaded}, None

    logger.debug("工具参数无法解析为 JSON：%r", raw[:200])
    return None, raw


#: Markdown 代码围栏（仅用于取出围栏内的裸 JSON，供数组分支使用）。
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _plain_candidates(raw: str) -> list[str]:
    """返回可直接交给 ``json.loads`` 的候选串（含围栏内内容）。"""
    text = strip_reasoning_tags(raw).strip()
    match = _FENCE_RE.search(text)
    return [text, match.group(1).strip() if match else ""]


def parse_tool_calls(raw_tool_calls: Any) -> tuple[ToolCall, ...]:
    """把后端返回的工具调用列表转换为领域模型。

    刻意对多种输入形态容错：SDK 对象、普通字典、以及 ``None``。
    不同后端（以及同一后端的不同版本）在这里的返回类型并不一致，
    把差异挡在本函数内部，上层就不必写三层 ``isinstance`` 判断。

    Args:
        raw_tool_calls: 后端返回的原始工具调用集合。

    Returns:
        解析后的 :class:`ToolCall` 元组。参数解析失败时 ``arguments`` 为空字典、
        ``raw_arguments`` 保留原文——调用方据此决定重试。
    """
    if not raw_tool_calls:
        return ()

    calls: list[ToolCall] = []
    for index, item in enumerate(raw_tool_calls):
        identifier, name, raw_arguments = _unwrap_tool_call(item, index)
        arguments, failure = parse_tool_arguments(raw_arguments)
        calls.append(
            ToolCall(
                id=identifier or f"call-{index}",
                name=name or "",
                arguments=arguments or {},
                raw_arguments=failure,
            )
        )
    return tuple(calls)


def _unwrap_tool_call(item: Any, index: int) -> tuple[str, str, str | None]:
    """从 SDK 对象或字典中取出 ``(id, name, 原始参数串)``。"""
    if isinstance(item, dict):
        function = item.get("function") or {}
        if isinstance(function, dict):
            return (
                str(item.get("id") or ""),
                str(function.get("name") or ""),
                function.get("arguments"),
            )
        return str(item.get("id") or ""), str(item.get("name") or ""), item.get("arguments")

    function = getattr(item, "function", None)
    identifier = str(getattr(item, "id", "") or "")
    if function is not None:
        return (
            identifier,
            str(getattr(function, "name", "") or ""),
            getattr(function, "arguments", None),
        )
    return identifier, str(getattr(item, "name", "") or ""), getattr(item, "arguments", None)


def to_backend_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """把领域消息转换为 OpenAI 兼容的消息字典。

    两处必须处理的细节：

    1. **assistant 的 tool_calls 必须原样回填**。多数提供商要求 assistant 消息中的
       工具调用与其后的 ``role="tool"`` 消息**成对出现**，缺失会直接返回 400；
    2. **推理模型的历史内容要剥离思维链**，否则上一轮的思考会被当作答案喂回去，
       既浪费 token 又会诱导模型重复思考。
    """
    payload: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role}

        if message.role == "assistant" and message.tool_calls:
            entry["content"] = strip_reasoning_tags(message.content) if message.content else ""
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            payload.append(entry)
            continue

        if message.role == "tool":
            entry["content"] = message.content
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            if message.name:
                entry["name"] = message.name
            payload.append(entry)
            continue

        entry["content"] = (
            strip_reasoning_tags(message.content)
            if message.role == "assistant"
            else message.content
        )
        payload.append(entry)
    return payload


def normalize_vector(vector: list[float]) -> list[float]:
    """把向量 L2 归一化。

    :mod:`scitrace.ports.llm` 约定嵌入向量必须归一化（索引层以点积实现余弦）。
    在**适配器边界**完成这一步，而不是指望每个后端都遵守约定：
    未归一化的向量不会报错，只会让相似度整体偏移，属于最难发现的一类错误。

    Args:
        vector: 原始向量。

    Returns:
        归一化后的向量。零向量原样返回（无法归一化，也不应伪造一个方向）。

    Raises:
        ValueError: 向量为空或含 NaN / 无穷值。
    """
    if not vector:
        raise ValueError("向量为空")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("向量含 NaN 或无穷值")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        logger.warning("收到零向量，无法归一化，将原样返回")
        return list(vector)
    return [value / norm for value in vector]

"""LLM 与 Embedding 客户端契约。

**自研抽象的理由**（SPEC §9 增量 ①）：直接依赖某个 LLM 编排框架会把
"消息格式""工具调用协议""用量统计口径"一并外包出去，导致 (a) 更换提供商时改动扩散，
(b) 无法精确控制成本闸门（增量 ④），(c) 引入与参考实现同族的依赖。
本项目的抽象只有四个 DTO 加两个 Protocol，实现可自由替换。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from scitrace.ports.common import ToolSpec

__all__ = [
    "EmbeddingClient",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "Role",
    "ToolCall",
]

Role = Literal["system", "user", "assistant", "tool"]


class LLMMessage(BaseModel):
    """一条对话消息。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = ""
    #: 当 ``role == "tool"`` 时，指向它所回应的那次工具调用。
    tool_call_id: str | None = None
    #: 当 ``role == "tool"`` 时的工具名，便于模型理解观测来源。
    name: str | None = None
    #: 当 ``role == "assistant"`` 且该轮发起了工具调用时，携带调用列表。
    #: 保留它是为了在多轮循环中原样回填历史——多数提供商要求 assistant 的
    #: tool_calls 与其后的 tool 消息成对出现，缺失会导致 400。
    tool_calls: tuple["ToolCall", ...] = ()


class ToolCall(BaseModel):
    """模型请求的一次工具调用。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: 参数 JSON 解析失败时的原始文本。非空即为一次 ``parse_failure``，
    #: 由 Agent 循环决定重试还是降级（SPEC §4.1）。
    raw_arguments: str | None = None


class LLMResponse(BaseModel):
    """一次 LLM 调用的结果。

    用量字段是**必需**的而非可选：成本可观测（增量 ④）要求每一次调用都被计量，
    如果允许实现方省略，缺口会出现在最需要的地方。
    """

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    #: 本次调用的估算成本。单位由 ``cost_currency`` 明确给出——
    #: 供应商多以人民币计价，把人民币数字塞进名叫 ``_usd`` 的字段是错的。
    cost: float = Field(default=0.0, ge=0.0)
    cost_currency: str = Field(default="USD", description="``cost`` 的币种")
    #: 命中提示词缓存的输入 token 数。缓存命中价通常便宜两个数量级，
    #: 忽略它会把成本高估数倍。
    cached_tokens: int = Field(default=0, ge=0)
    cost_known: bool = Field(
        default=True,
        description=(
            "成本是否可信。模型不在任何价格表中时为 False，此时 ``cost`` 只是下界。"
            "**不允许静默记 0**：那会让上层的成本闸门看起来在工作而从不触发。"
        ),
    )
    finish_reason: str = ""


@runtime_checkable
class LLMClient(Protocol):
    """文本生成客户端。

    实现方需保证：``complete`` 在失败时抛出异常而不是返回空响应——
    静默的空响应会被上层当作"模型认为没有答案"，从而污染拒答判定。
    """

    @property
    def model_name(self) -> str:
        """模型标识，用于用量报告与索引/配置指纹。"""
        ...

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_object: bool = False,
    ) -> LLMResponse:
        """生成一次回复。

        Args:
            messages: 对话历史。
            tools: 可用工具；为 ``None`` 时不启用工具调用。
            temperature: 采样温度；``None`` 表示用提供商默认值。
            max_tokens: 输出长度上限。
            json_object: 是否要求返回合法 JSON 对象。
                **仅作提示**：并非所有提供商都支持，且模型仍可能返回非法 JSON，
                因此调用方必须保留解析容错链（SPEC §3.7）。

        Returns:
            模型回复与用量。
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """文本向量化客户端。"""

    @property
    def model_name(self) -> str:
        """模型标识。参与索引指纹计算（SPEC §5.3）——换模型必须导致索引重建。"""
        ...

    @property
    def dimension(self) -> int:
        """向量维度。首次调用后必须可用，用于校验索引与模型是否匹配。"""
        ...

    async def embed(
        self, texts: list[str], *, kind: Literal["query", "document"] = "document"
    ) -> list[list[float]]:
        """把文本编码为 **L2 归一化**的向量。

        ``kind`` 用于支持查询/文档非对称编码的模型（如 BGE 系列需要给查询加指令前缀）。
        实现方若使用对称模型，可忽略该参数。

        归一化是必需的：向量索引以点积实现余弦相似度，未归一化会让 MMR
        （SPEC §3.6）中的距离比较失去一致量纲。

        Args:
            texts: 待编码文本，非空。
            kind: 该批文本是查询还是文档。

        Returns:
            与 ``texts`` 等长的向量列表。
        """
        ...

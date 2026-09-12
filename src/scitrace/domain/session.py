"""会话状态：一次问答的完整记录。

``Session`` 是"可复现"与"可审计"两项要求的载体。它被持久化到
``~/.scitrace/sessions/<fingerprint>/<session_id>.json``，使得任何一次历史回答都能被
重新检查：用了哪些证据、引用了哪些片段、花了多少 token、走了哪些工具调用。

**为什么不复用日志做这件事**：日志是给人看的、格式不稳定；会话记录是给机器校验的
（SPEC §1.3 "引用可回溯率 100%" 需要机器逐条核对每个引用键）。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scitrace.domain.citation import Citation
from scitrace.domain.evidence import Evidence
from scitrace.util.sanitize import SanitizedModel
from scitrace.util.hashing import derive_key

__all__ = [
    "ActionRecord",
    "Answer",
    "Session",
    "SessionStatus",
    "StageTiming",
    "Usage",
]


class SessionStatus(StrEnum):
    """会话终态。取值语义与 SPEC §4.2 的状态机一一对应。"""

    #: 拿到答案且证据充分。
    SUCCESS = "SUCCESS"
    #: 正常结束，但模型自述无法确定（``finish(has_answer=false)``）。
    UNSURE = "UNSURE"
    #: 因超时/步数上限/预算触顶而被强制收尾，答案基于已有证据。
    TRUNCATED = "TRUNCATED"
    #: 不可恢复错误。
    FAIL = "FAIL"
    #: 索引为空或证据不足，明确拒答。**这是正常终态，不是错误。**
    REFUSED = "REFUSED"
    #: 模型给出了答案，但**没有引用任何有效证据**。
    #:
    #: 与 ``REFUSED`` 必须分开：``REFUSED`` 是"证据不足"这一**正常**结果，
    #: 而 ``UNCITED`` 意味着模型违反了引用约束——它需要的是改提示词或换模型，
    #: 不是接受。把两者合并成一个状态，使用方就无法区分"系统说不知道"
    #: 与"系统给了没有依据的答案"。
    UNCITED = "UNCITED"


class Usage(SanitizedModel):
    """token 用量与成本统计。【原创增量 ④ 的可观测基础】

    所有计数默认 0，且 :meth:`merge` 返回新对象——累计用量在多处并发发生
    （并发筛选、多轮 Agent），用不可变累加避免共享可变状态带来的竞态。
    """

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(
        default=0,
        ge=0,
        description=(
            "``prompt_tokens`` 中命中提示词缓存的部分。它**已包含在** ``prompt_tokens`` 内，"
            "单独记是因为缓存命中价通常便宜两个数量级——不区分会让成本高估数倍。"
        ),
    )
    estimated_cost: float = Field(default=0.0, ge=0.0)
    cost_currency: str = Field(default="USD", description="``estimated_cost`` 的币种")
    cost_known: bool = Field(
        default=True,
        description=(
            "本次会话的成本是否**可信**。False 表示至少有一次调用的模型不在价格表中，"
            "``estimated_cost`` 只是下界（通常为 0）。"
            "**必须显式标记**：把未知成本静默记成 0，会让成本闸门看起来在工作而实际从不触发——"
            "这是本项目实测踩过的坑（deepseek-v4-flash 不在 litellm 价格表中）。"
        ),
    )
    llm_calls: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    parse_failures: int = Field(
        default=0,
        ge=0,
        description="筛选阶段 LLM 输出无法解析为结构化结果的次数（SPEC §3.7 容错链）",
    )
    dangling_citations: int = Field(
        default=0,
        ge=0,
        description="答案中出现但无法解析到本次证据集的引用键数量，理想值恒为 0",
    )
    metadata_incomplete: int = Field(
        default=0, ge=0, description="引用渲染时缺少元数据、退化为 unknown 的文献数"
    )

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: Self) -> Self:
        """累加两份用量，返回新对象。

        ``cost_known`` 是**布尔**而非计数，用逻辑与合并：
        只要有一份用量的成本不可信，合并结果就不可信。把它也当计数相加会得到
        一个恒为真的整数，让标记失去意义。

        币种不是可加量，取哪一边需要判断。**不能无条件取 ``self``**：
        本方法最常见的用法是"把一个空累加器与一次真实读数合并"
        （``Usage().merge(读数)``），而空累加器的币种只是字段默认值 ``USD``，
        没有任何信息量。无条件取 ``self`` 会让一次人民币计价的会话
        全程显示 ``cost_currency="USD"``——**数值是人民币，标签写美元**。
        实测踩到过：agentic 会话报出 ``0.279 USD``，而按配置它其实是 0.279 CNY。

        规则：**空累加器（identity）让位于有读数的一方**；两边都有读数时取
        ``self``（同一次会话内计价配置相同，正常情形下两者相等）。
        """
        merged = {
            name: getattr(self, name) + getattr(other, name)
            for name in type(self).model_fields
            if name not in {"cost_known", "cost_currency"}
        }
        merged["cost_known"] = self.cost_known and other.cost_known
        merged["cost_currency"] = (
            other.cost_currency if self._should_defer_currency(other) else self.cost_currency
        )
        return type(self)(**merged)

    def _should_defer_currency(self, other: Self) -> bool:
        """合并时是否应把币种让给 ``other``。

        两种情形需要让位，其余情形由 ``self`` 持有（同一次会话内计价配置相同）：

        1. ``self`` 是累加单位元——它还没并入任何读数，币种只是字段默认值；
        2. ``other`` 知道自己的成本而 ``self`` 不知道——``cost_known`` 为假时
           币种没有意义，此时有读数的一方才是币种的来源。
        """
        return self.is_identity or (other.cost_known and not self.cost_known)

    @property
    def is_identity(self) -> bool:
        """是否是累加单位元（所有计数为零，尚未并入任何读数）。"""
        return all(
            getattr(self, name) == 0
            for name in type(self).model_fields
            if name not in {"cost_currency", "cost_known"}
        )

    def since(self, baseline: Self) -> Self:
        """相对某个基线快照的增量，用于把"进程累计用量"切成"单次会话用量"。

        ``Services.usage`` 是**跨多次** :func:`~scitrace.api.ask` 持续累加的累加器。
        直接把它当作某一次会话的用量会同时坏掉两件事：

        1. 评测算出的"每题成本/token/调用数"其实是**累计值**；
        2. ``agent.max_cost`` 这个**会话级**预算变成"整个进程的预算"——
           实测后果：同一进程里第一题烧穿预算后，后续每一题都在进门时
           被判超支，直接拒答，且 ``actions == 0``（一次工具都没调）。
           一整轮 9 题评测因此全部作废。

        Args:
            baseline: 本次会话开始前的用量快照。

        Returns:
            两者之差；币种取自身的，``cost_known`` 取两者的逻辑与。
        """
        data = {
            name: getattr(self, name) - getattr(baseline, name)
            for name in type(self).model_fields
            if name not in {"cost_known", "cost_currency"}
        }
        data["cost_known"] = self.cost_known and baseline.cost_known
        data["cost_currency"] = self.cost_currency
        return type(self)(**data)


class StageTiming(SanitizedModel):
    """各阶段耗时（秒），用于定位性能瓶颈。"""

    ingest_s: float = Field(default=0.0, ge=0.0)
    retrieve_s: float = Field(default=0.0, ge=0.0)
    screen_s: float = Field(default=0.0, ge=0.0)
    synthesize_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)


class ActionRecord(SanitizedModel):
    """一次工具调用的记录。

    只保存观测结果的**摘要**而非全文：完整观测可能包含数十个片段的原文，
    全量落盘会让会话文件膨胀到 MB 级，而其内容已可从证据集还原。
    """

    step: int = Field(ge=0)
    tool: str
    arguments: dict[str, object] = Field(default_factory=dict)
    observation_summary: str = ""
    #: 本步执行**之后**的累计 token。相邻两步之差即该步的净消耗。
    #:
    #: 加这个字段的直接动机：Agentic 模式唯一一次真实运行烧掉 171,694 token，
    #: 而当时的会话记录里**看不到 token 花在哪一步**——只能看到一个总数。
    #: 没有逐步分解，任何关于"成本从哪来"的判断都只能是猜测。
    tokens_after: int = Field(default=0, ge=0)
    #: 回灌给模型的观测**全文**长度（``observation_summary`` 是截断到 300 字后的版本，
    #: 只用于会话记录）。它直接决定历史膨胀的速度。
    observation_chars: int = Field(default=0, ge=0)
    duration_s: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class Answer(SanitizedModel):
    """最终答案及其引用解析结果。"""

    text: str = Field(description="已把引用键替换为可读文内引用的答案正文")
    raw_text: str = Field(default="", description="模型原始输出，保留以便排障与审计")
    citations: list[Citation] = Field(default_factory=list)
    references: dict[str, str] = Field(
        default_factory=dict, description="reference_key -> BibTeX 条目，按首次引用顺序"
    )
    refused: bool = False
    refusal_reason: str = ""
    uncited: bool = Field(
        default=False,
        description=(
            "模型给出了答案正文，但没有引用任何有效证据。"
            "与 ``refused`` 的区别在于**原因**：拒答是「证据不足」这一正常结果，"
            "而本条是模型违反了引用约束。二者的处置不同（前者接受，后者要改提示词或换模型），"
            "因此不能合并成一个布尔量。"
        ),
    )

    @property
    def has_citations(self) -> bool:
        """答案是否带至少一处引用。"""
        return bool(self.citations)


class Session(SanitizedModel):
    """一次问答的完整状态。"""

    session_id: str
    question: str
    fingerprint: str = Field(description="索引指纹，标识本次问答所用索引")
    mode: Literal["agentic", "deterministic"] = "deterministic"
    status: SessionStatus = SessionStatus.FAIL
    answer: Answer | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    timing: StageTiming = Field(default_factory=StageTiming)
    actions: list[ActionRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: list[str] = Field(
        default_factory=list, description="降级、告警等运行期说明（如 'budget_exceeded'）"
    )

    @field_validator("question")
    @classmethod
    def _require_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Session.question 不能为空")
        return cleaned

    @classmethod
    def new(
        cls,
        *,
        question: str,
        fingerprint: str,
        mode: Literal["agentic", "deterministic"] = "deterministic",
    ) -> Self:
        """创建一个新会话，``session_id`` 由问题与指纹确定性派生。

        确定性 id 使得"同问题同索引重跑"会覆盖同一条记录，而不是堆积
        语义重复的历史——这对回归测试与消融实验（需要稳定可比对的结果）是必要的。
        """
        return cls(
            session_id=derive_key(question, fingerprint, length=12),
            question=question,
            fingerprint=fingerprint,
            mode=mode,
        )

    def get_evidence(self, key: str) -> Evidence | None:
        """按引用键查找证据。"""
        return next((item for item in self.evidence if item.key == key), None)

    def to_json(self) -> str:
        """序列化为 JSON 文本（保留非 ASCII，便于人工阅读中文答案）。"""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """从 JSON 文本反序列化。"""
        return cls.model_validate(json.loads(payload))

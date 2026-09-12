"""Agent 循环与状态机（SPEC §4.2 / §4.3）。

## 两种模式的意义

- ``deterministic``：固定执行 检索 → 取证 → 合成 → 结束，不调用模型做工具选择。
  同输入同配置**字节级可复现**。它是回归测试与消融实验的基线——
  没有它，"Agentic 到底比固定流程好多少"就无法回答。
- ``agentic``：由模型自主决定工具与顺序。

## 兜底为什么必须存在

Agentic 流程的三个终止条件（步数、超时、预算）都不是异常，而是**常态**：
复杂问题会让模型反复检索，一次问答花掉几十万 token 是很容易发生的事。
关键在于终止时的动作是**用已有证据强制收尾**，而不是抛错丢弃工作——
一次问答已经消耗的成本不该因为超了 5% 的预算就全部作废。

超时的实现有一个容易写错的地方：**合成必须放在超时作用域之外**，
否则"超时"会导致连最后一次合成也被取消，用户拿到的是一个空答案。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import anyio

from scitrace.agent.budget import Budget
from scitrace.agent.state import AgentState, Tool, ToolOutcome, status_line
from scitrace.agent.tools import build_default_tools
from scitrace.domain import Answer, Session, SessionStatus
from scitrace.domain.session import ActionRecord, StageTiming, Usage
from scitrace.ports import LLMMessage
from scitrace.service import SessionStore

logger = logging.getLogger(__name__)

__all__ = ["AgentRunResult", "AgentRuntime"]

#: agentic 模式下系统提示词。说明可用工具与工作方式。
#:
#: 与筛选/合成提示词一样，本段文字由本项目撰写（见 ``prompts`` 的撰写纪律）。
_AGENT_SYSTEM_PROMPT = """\
你是一名科研文献研究助手。你的任务是回答问题，且**每一处结论都必须有文献证据支撑**。

你可以使用以下工具：
- `search_literature`：检索相关论文，确定候选论文集。
- `gather_evidence`：在候选论文范围内收集证据。
- `answer_question`：基于已有证据生成带引用的答案（不结束会话）。
- `reset_scope`：清空候选论文集，保留已收集的证据。
- `finish`：结束会话。证据不足时必须把 has_answer 设为 false，不要给出猜测。

工作要求：
1. 先检索、再取证、最后作答。不要在没有任何证据的情况下调用 answer_question。
2. 如果一次检索的结果不理想，换用更具体或更宽泛的查询再试，而不是直接放弃。
3. 连续两次取证都没有新增证据时，说明该范围已经挖尽：换一个角度，
   或者用已有证据作答并结束。
4. 工具返回的状态串（papers / evidence / cost）反映了当前进度，据此判断是否应当收手。
"""


@dataclass
class AgentRunResult:
    """一次 Agent 运行的完整结果。"""

    question: str
    answer: Answer
    status: SessionStatus
    actions: list[ActionRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    timing: StageTiming = field(default_factory=StageTiming)
    notes: list[str] = field(default_factory=list)
    session_id: str = ""


def _refusal(reason: str) -> Answer:
    """构造一个拒答答案。"""
    return Answer(text="", raw_text="", refused=True, refusal_reason=reason)


class AgentRuntime:
    """把一次问答跑完。

    构造后调用 :meth:`run`；运行结束时会话会被持久化到
    ``settings.sessions_dir``，便于事后审计"这次答案用了哪些证据"。
    """

    def __init__(
        self,
        *,
        services,  # noqa: ANN001 - factory.Services，避免循环导入
        question: str,
        mode: Literal["agentic", "deterministic"] = "deterministic",
        on_step: Callable[[ActionRecord], Awaitable[None]] | None = None,
    ) -> None:
        self.services = services
        self.question = question
        self.mode = mode
        self.on_step = on_step
        self.settings = services.settings
        self.tools: list[Tool] = build_default_tools(services)
        self._by_name = {tool.name: tool for tool in self.tools}
        self.state = AgentState(question=question)
        self.budget = Budget(
            max_tokens=self.settings.agent.max_tokens,
            max_cost=self.settings.agent.max_cost,
        )
        # 配了成本上限、却拿不到可信成本 → 闸门实际上不生效。
        # 这种情况必须明确告知，否则用户会以为成本受到管控。
        if self.budget.cost_gate_active and not self.services.usage.cost_known:
            self.state.notes.append("cost_gate_inactive")
        self.timing = StageTiming()
        self._step = 0

    # ---------------------------------------------------------------- 入口 --

    async def run(self) -> AgentRunResult:
        """执行一次完整的问答。"""
        started = time.perf_counter()
        status = SessionStatus.FAIL
        try:
            if self.mode == "deterministic":
                status = await self._run_deterministic()
            else:
                status = await self._run_agentic()
        except Exception:  # noqa: BLE001 - 任何未预期错误都要产出一个结果而非抛给上层
            logger.exception("Agent 运行出现未预期错误")
            status = SessionStatus.FAIL
            self.state.notes.append("unexpected_error")

        self.timing.total_s = time.perf_counter() - started
        answer = self.state.answer or _refusal("未能生成答案")
        result = AgentRunResult(
            question=self.question,
            answer=answer,
            status=status,
            actions=list(self.state.actions),
            usage=self.services.usage,
            timing=self.timing,
            notes=list(self.state.notes),
        )
        result.session_id = self._persist(result)
        logger.info(
            "问答结束：状态=%s 证据=%d 引用=%d 耗时=%.1fs 成本=$%.4f",
            status,
            len(self.state.evidence),
            len(answer.citations),
            self.timing.total_s,
            result.usage.estimated_cost,
        )
        return result

    # ------------------------------------------------------------ 两种模式 --

    async def _run_deterministic(self) -> SessionStatus:
        """固定顺序执行，不调用模型做工具选择。

        **字节级可复现**：不使用随机数、不依赖并发完成顺序、不读当前时间
        （时间只进入 ``timing``，不进入 ``actions`` 或答案）。
        """
        await self._execute("search_literature", {"query": self.question})
        await self._execute("gather_evidence", {"question": self.question})
        if not self.state.evidence:
            self.state.notes.append("no_evidence")
            self.state.answer = _refusal("没有检索到可用的证据")
            return SessionStatus.REFUSED
        await self._execute("answer_question", {})
        await self._execute("finish", {"has_answer": True, "reason": "确定性流程完成"})
        return self._final_status(fallback=SessionStatus.SUCCESS)

    async def _run_agentic(self) -> SessionStatus:
        """由模型决定工具与顺序。"""
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=_AGENT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=f"问题：{self.question}"),
        ]
        specs = [tool.spec() for tool in self.tools]
        no_new_evidence = 0
        param_failures = 0
        termination: str | None = None

        # 墙钟超时只覆盖"模型决策 + 工具执行"的循环；
        # 强制收尾的合成在作用域之外执行（见模块 docstring）。
        with anyio.move_on_after(self.settings.agent.timeout_seconds) as scope:
            while self._step < self.settings.agent.max_steps:
                if self.budget.exhausted(self.services.usage):
                    termination = "budget_exceeded"
                    break
                response = await self.services.llm("agent").complete(messages, tools=specs)
                self.services.merge_usage(
                    Usage(
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        estimated_cost=response.cost,
                        llm_calls=1,
                    )
                )
                if not response.tool_calls:
                    # 模型没调用工具而是直接作答：把它当作最终答案来源
                    self.state.notes.append("model_answered_directly")
                    break

                messages.append(
                    LLMMessage(
                        role="assistant", content=response.content, tool_calls=response.tool_calls
                    )
                )
                finished = False
                for call in response.tool_calls:
                    if call.raw_arguments is not None:
                        param_failures += 1
                        if param_failures > self.settings.agent.max_tool_param_retries:
                            termination = "tool_param_retries_exceeded"
                            finished = True
                            break
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=call.id,
                                name=call.name,
                                content=(
                                    f"参数不是合法 JSON：{call.raw_arguments[:200]}。"
                                    "请重新给出结构正确的参数。"
                                ),
                            )
                        )
                        continue
                    before = len(self.state.evidence)
                    outcome = await self._execute(call.name, call.arguments)
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=call.id,
                            name=call.name,
                            content=outcome.observation,
                        )
                    )
                    if call.name == "gather_evidence":
                        no_new_evidence = no_new_evidence + 1 if len(self.state.evidence) == before else 0
                    if outcome.stop:
                        finished = True
                        break
                if finished:
                    break
                if no_new_evidence >= self.settings.agent.no_new_evidence_limit:
                    termination = "no_new_evidence"
                    break
            else:
                termination = "max_steps_exceeded"

        if scope.cancelled_caught:
            termination = "timeout"
        if termination:
            self.state.notes.append(termination)

        # 强制收尾（**在超时作用域之外**，因此超时也一定拿得到答案）
        if not self.state.evidence:
            self.state.notes.append("no_evidence")
            self.state.answer = _refusal("没有检索到可用的证据")
            return SessionStatus.REFUSED
        if self.state.answer is None or termination is not None:
            await self._execute("answer_question", {})
        return self._final_status(
            fallback=SessionStatus.SUCCESS if self.state.answer else SessionStatus.FAIL
        )

    # ------------------------------------------------------------ 工具执行 --

    async def _execute(self, name: str, arguments: dict[str, object]) -> ToolOutcome:
        """执行一个工具并记录动作。"""
        tool = self._by_name.get(name)
        if tool is None:
            return ToolOutcome(observation=f"未知工具 {name!r}。可用工具：{', '.join(self._by_name)}")

        self._step += 1
        started = time.perf_counter()
        outcome = await tool.run(arguments, self.state)
        duration = time.perf_counter() - started

        record = ActionRecord(
            step=self._step,
            tool=name,
            arguments=dict(arguments),
            observation_summary=outcome.observation[:300],
            duration_s=duration,
        )
        self.state.actions.append(record)
        if self.on_step is not None:
            await self.on_step(record)
        logger.debug("步骤 %d：%s（%.2fs）", self._step, name, duration)
        return outcome

    # -------------------------------------------------------------- 收尾 --

    def _final_status(self, *, fallback: SessionStatus) -> SessionStatus:
        """按状态机的优先级确定终态。

        优先级：被截断 > 未确定 > 兜底。截断优先是因为它是对用户最需要知道的
        事实——"这个答案不完整"比"这个答案不够确定"更重要。
        """
        if "synthesis_failed" in self.state.notes:
            # 合成故障优先于一切：用户拿到的是"没有答案"，而不是"不确定的答案"。
            return SessionStatus.FAIL
        answer = self.state.answer
        if answer is not None and answer.uncited:
            # 模型答了但没引用：这**不是**拒答，而是模型违反了引用约束。
            # 单独成态，使用方才能区分"系统说不知道"与"系统给了无据的答案"。
            return SessionStatus.UNCITED
        for reason in ("timeout", "budget_exceeded", "max_steps_exceeded", "no_new_evidence"):
            if reason in self.state.notes:
                return SessionStatus.TRUNCATED
        if any(note.startswith("finish:") for note in self.state.notes):
            # finish 工具表达了 has_answer；为 false 时状态是 UNSURE
            return SessionStatus.SUCCESS if self.state.answer and not self.state.answer.refused else SessionStatus.UNSURE
        answer = self.state.answer
        if answer is not None and answer.refused:
            return SessionStatus.REFUSED
        return fallback

    def _persist(self, result: AgentRunResult) -> str:
        """把会话写入历史，返回 ``session_id``。"""
        try:
            session = Session.new(
                question=self.question,
                fingerprint=self.settings.index_fingerprint(),
                mode=self.mode,
            )
            session.status = result.status
            session.answer = result.answer
            session.evidence = list(self.state.evidence)
            session.usage = result.usage
            session.timing = result.timing
            session.actions = result.actions
            session.notes = result.notes
            SessionStore(self.settings.sessions_dir).append(session)
            return session.session_id
        except Exception as error:  # noqa: BLE001 - 持久化失败不该让问答失败
            logger.warning("会话持久化失败（不影响本次结果）：%s", error)
            return ""

    def status_snapshot(self) -> str:
        """当前状态串，供调试与流式输出使用。"""
        return status_line(
            self.state,
            cost=self.services.usage.estimated_cost,
            currency=self.services.usage.cost_currency,
        )

"""测试 Agent 运行时与 SPEC §8 的可观测行为契约。

本文件覆盖两类内容：

1. **Agent 的终止条件**——SPEC §4.3 列出的每一种兜底都必须在测试里出现一次。
   这些分支的共同点是"平时不会走到，一旦走到就必须正确"，而它们恰恰是最难
   在手工验证中触发的。
2. **行为契约 C1–C8**——"功能与逻辑对齐参考实现"这句话如果不是一句空话，
   就必须有可执行的定义。C1–C8 就是那个定义。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import (
    FakeEmbedder,
    FakeFullTextIndex,
    FakeLLMClient,
    FakeScreener,
    FakeVectorIndex,
)
from scitrace.agent.budget import Budget
from scitrace.agent.runtime import AgentRuntime
from scitrace.agent.state import AgentState, status_line
from scitrace.agent.tools import build_default_tools
from scitrace.config import AnswerSettings, RetrievalSettings, Settings
from scitrace.domain import Evidence, Source, make_evidence_key
from scitrace.domain.session import SessionStatus, Usage
from scitrace.factory import Services
from scitrace.pipeline.retrieval import Retriever
from scitrace.pipeline.synthesis import AnswerSynthesizer
from scitrace.prompts import INSUFFICIENT_EVIDENCE, get_prompt_set
from scitrace.service import SessionStore


def make_source(key: str = "src-1", *, title: str = "证据可溯源问答方法研究") -> Source:
    return Source(
        key=key,
        content_hash="h" * 16,
        rel_path=f"corpus/{key}.txt",
        title=title,
        authors=["张三"],
        year=2024,
        doi="10.5555/example.2024.001",
    )


class FakeScreenerWithEvidence(FakeScreener):
    """无条件为每个片段产出证据，便于构造"有证据"的路径。"""

    def __init__(self, *, min_relevance: int = 0, score: int = 9, **kwargs) -> None:
        super().__init__(**kwargs)
        # 端口明确要求筛选器"已应用 screening.min_relevance 阈值"，
        # 因此替身必须实现它——否则测试会验证一个现实中不存在的宽松行为。
        self.min_relevance = min_relevance
        self.score = score
        self.last_usage = Usage(llm_calls=1, prompt_tokens=5, completion_tokens=5)

    async def screen(self, question, fragments, *, sources):  # noqa: ANN001
        results = await super().screen(question, fragments, sources=sources)
        if not results and fragments:
            # 片段太短也会产出证据：测试关心的是流程，不是长度启发式
            item = fragments[0]
            source = sources.get(item.source_key)
            stem = source.citation_stem if source else "unknown"
            results = [
                Evidence(
                    key=make_evidence_key(item.source_key, item.fragment_id),
                    source_key=item.source_key,
                    fragment_id=item.fragment_id,
                    summary=item.text[:100] or "摘要",
                    relevance=self.score,
                    citation=f"({stem} {item.page_label})".replace(" )", ")"),
                    page_label=item.page_label,
                )
            ]
        self.last_usage = Usage(llm_calls=1, prompt_tokens=5, completion_tokens=5)
        # 与真实实现一致：过滤、降序、截断
        kept = [item for item in results if item.relevance >= self.min_relevance]
        kept.sort(key=lambda item: (-item.relevance, item.key))
        return kept


async def looping_llm(*, tool: str = "gather_evidence", arguments: dict | None = None,
                      delay: float = 0.0):
    """构造一个**永不主动结束**的模型替身。

    终止条件的测试必须有一个不会自己停下来的模型——否则循环会因
    "模型直接作答"而正常退出，根本走不到兜底分支。
    """
    import anyio as _anyio

    from scitrace.ports import LLMResponse, ToolCall

    async def complete(messages, **kwargs):  # noqa: ANN001
        if delay:
            await _anyio.sleep(delay)
        return LLMResponse(
            content="",
            tool_calls=(
                ToolCall(
                    id="c1",
                    name=tool,
                    arguments=arguments if arguments is not None else {"question": "Q"},
                ),
            ),
            prompt_tokens=1,
            completion_tokens=1,
        )

    return complete


class FakeServicesFactory:
    """构造一套用替身装配的 Services。"""

    def __init__(self, tmp_path: Path, *, answer_text: str | None = None, seed: bool = True):
        self.tmp_path = tmp_path
        settings = Settings()
        settings.index.root = tmp_path / "index"
        settings.index.sessions_root = tmp_path / "sessions"
        settings.retrieval = RetrievalSettings(strategy="dense", k=5)
        settings.answer = AnswerSettings(max_evidence=5)
        self.settings = settings

        self.vector = FakeVectorIndex()
        self.fulltext = FakeFullTextIndex()
        self.embedder = FakeEmbedder(dimension=8)
        self.sources = {"src-1": make_source()}
        self.seeded = seed

        self.llm = FakeLLMClient(
            [answer_text] if answer_text else [f"结论是 X ({self._key()})。"]
        )
        self.llm_summary = FakeLLMClient()
        # agent 角色用**独立**的客户端对象。生产里三个角色是三个客户端实例
        # （可能连模型都不同），测试里共用同一个对象会让"覆盖 agent 行为"
        # 顺带改掉合成行为，从而掩盖真实的失败模式。
        self.llm_agent = FakeLLMClient()
        self.screener = FakeScreenerWithEvidence()

    def _key(self) -> str:
        return make_evidence_key("src-1", "frag000")

    async def build(self) -> Services:
        from scitrace.domain import Fragment

        if self.seeded:
            fragment = Fragment(
                fragment_id="frag000",
                source_key="src-1",
                text="本文报告的精确率为 85.2%，显著高于基线方法。",
                chunk_index=0,
                page_start=3,
                page_end=4,
                char_count=24,
            )
            vectors = await self.embedder.embed([fragment.text])
            await self.vector.add([(fragment, vectors[0])])
            await self.fulltext.add([fragment], titles={"src-1": self.sources["src-1"].title})
            await self.fulltext.commit()

        retriever = Retriever(
            vector_index=self.vector,
            fulltext_index=self.fulltext,
            embedder=self.embedder,
            settings=self.settings.retrieval,
        )
        synthesizer = AnswerSynthesizer(
            llm=self.llm, prompts=get_prompt_set("zh"), settings=self.settings.answer
        )
        return Services(
            settings=self.settings,
            prompts=get_prompt_set("zh"),
            vector_index=self.vector,  # type: ignore[arg-type]
            fulltext_index=self.fulltext,  # type: ignore[arg-type]
            embedder=self.embedder,
            llms={"main": self.llm, "summary": self.llm_summary, "agent": self.llm_agent},
            retriever=retriever,
            screener=self.screener,
            synthesizer=synthesizer,
            sources=self.sources,
        )


class TestAgentState:
    def test_add_evidence_deduplicates(self) -> None:
        state = AgentState(question="q")
        item = Evidence(
            key="ev-1", source_key="s", fragment_id="f", summary="x", relevance=8
        )
        assert state.add_evidence([item]) == 1
        assert state.add_evidence([item]) == 0
        assert len(state.evidence) == 1

    def test_status_line_distinguishes_unscoped_from_empty(self) -> None:
        """``None``（尚未限定范围）与空集合（限定了但没找到）语义不同，不能合并。"""
        unscoped = AgentState(question="q")
        empty = AgentState(question="q", scoped_source_keys=set())
        assert "全部" in status_line(unscoped)
        assert "0/0" in status_line(empty)


class TestBudget:
    def test_unlimited(self) -> None:
        assert Budget().check(Usage(prompt_tokens=10**9)) is None

    def test_exactly_at_limit_is_not_exceeded(self) -> None:
        """用到上限是正常结束，不是超限。"""
        assert Budget(max_tokens=100).check(Usage(prompt_tokens=100)) is None
        assert Budget(max_cost=1.0).check(Usage(estimated_cost=1.0)) is None

    def test_float_tolerance(self) -> None:
        """浮点累加误差不该触发"明明刚好用完却报超限"。"""
        budget = Budget(max_cost=0.3)
        assert budget.check(Usage(estimated_cost=0.1 + 0.2)) is None

    def test_token_over_limit(self) -> None:
        assert Budget(max_tokens=100).check(Usage(prompt_tokens=101)) is not None

    def test_cost_over_limit(self) -> None:
        assert Budget(max_cost=1.0).check(Usage(estimated_cost=1.5)) is not None

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            Budget(max_tokens=0)
        with pytest.raises(ValueError):
            Budget(max_cost=-1)


class TestTools:
    async def test_specs_have_json_schema(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        for tool in build_default_tools(services):
            spec = tool.spec()
            assert spec.name
            assert spec.description
            assert spec.parameters.get("type") == "object"

    async def test_invalid_arguments_return_observation_not_exception(
        self, tmp_path: Path
    ) -> None:
        """模型给出非法参数是常态，为此中断会话等于把可恢复的小失误放大成失败。"""
        services = await FakeServicesFactory(tmp_path).build()
        tool = next(t for t in build_default_tools(services) if t.name == "search_literature")
        outcome = await tool.run({"wrong_field": 1}, AgentState(question="q"))
        assert not outcome.stop
        assert "不合法" in outcome.observation

    async def test_unknown_tool_is_reported(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        runtime = AgentRuntime(services=services, question="q")
        outcome = await runtime._execute("no_such_tool", {})
        assert "未知工具" in outcome.observation

    async def test_finish_marks_stop(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        tool = next(t for t in build_default_tools(services) if t.name == "finish")
        outcome = await tool.run({"has_answer": True, "reason": "done"}, AgentState(question="q"))
        assert outcome.stop is True
        assert outcome.has_answer is True

    async def test_answer_without_evidence_is_refused(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        tool = next(t for t in build_default_tools(services) if t.name == "answer_question")
        outcome = await tool.run({}, AgentState(question="q"))
        assert "没有任何证据" in outcome.observation


class TestDeterministicMode:
    async def test_produces_answer_with_citation(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        result = await AgentRuntime(services=services, question="精确率是多少？").run()

        assert result.status is SessionStatus.SUCCESS
        assert result.answer.text
        assert result.answer.citations, "成功答案必须带可回溯引用（契约 C2）"
        assert result.answer.references

    async def test_is_byte_level_reproducible(self, tmp_path: Path) -> None:
        """同输入同配置必须字节级可复现——这是回归测试与消融实验的前提。

        时间只进入 timing，不进入 actions 与答案正文。
        """
        first = await AgentRuntime(
            services=await FakeServicesFactory(tmp_path).build(), question="Q"
        ).run()
        second = await AgentRuntime(
            services=await FakeServicesFactory(tmp_path).build(), question="Q"
        ).run()
        assert [a.tool for a in first.actions] == [a.tool for a in second.actions]
        assert first.answer.text == second.answer.text
        assert first.answer.raw_text == second.answer.raw_text

    async def test_stage_order_is_search_then_gather_then_answer(self, tmp_path: Path) -> None:
        """契约 C1：一次成功问答必然经过 检索 → 取证 → 合成 三阶段。"""
        services = await FakeServicesFactory(tmp_path).build()
        result = await AgentRuntime(services=services, question="Q").run()
        tools = [action.tool for action in result.actions]
        assert tools.index("search_literature") < tools.index("gather_evidence")
        assert tools.index("gather_evidence") < tools.index("answer_question")

    async def test_no_evidence_refuses_without_calling_model(self, tmp_path: Path) -> None:
        """契约 C3：语料外问题必须拒答，且不得调用合成模型。

        空上下文下让模型作答，等于请它凭记忆编造。
        """
        factory = FakeServicesFactory(tmp_path, seed=False)
        services = await factory.build()
        result = await AgentRuntime(services=services, question="库外问题").run()

        assert result.status is SessionStatus.REFUSED
        assert result.answer.refused is True
        assert result.answer.citations == []
        assert factory.llm.calls == []

    async def test_usage_is_accumulated_across_stages(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        result = await AgentRuntime(services=services, question="Q").run()
        # 筛选与合成各至少一次调用
        assert result.usage.llm_calls >= 2
        assert result.usage.total_tokens > 0

    async def test_reported_usage_is_per_session_not_process_cumulative(
        self, tmp_path: Path
    ) -> None:
        """**回归测试**：``result.usage`` 必须是**本次会话**的用量。

        ``Services.usage`` 跨多次 ``ask()`` 持续累加。直接把它当会话用量，
        评测算出的"每题成本/token/调用数"就全是累计值——
        实测一整轮 9 题评测因此作废（每题都报同一个递增数）。
        """
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        first = await AgentRuntime(services=services, question="Q").run()
        second = await AgentRuntime(services=services, question="Q").run()

        assert first.usage.llm_calls > 0
        assert second.usage.llm_calls == first.usage.llm_calls, "第二次报的是累计值"
        assert second.usage.total_tokens == first.usage.total_tokens
        # 进程级累加器仍然如实保留总量，会话用量只是它的切片
        assert services.usage.llm_calls == first.usage.llm_calls + second.usage.llm_calls
        assert services.usage.total_tokens == first.usage.total_tokens * 2

    async def test_budget_applies_per_session_not_to_the_whole_process(
        self, tmp_path: Path
    ) -> None:
        """**回归测试**：``agent.max_tokens`` 是**会话级**预算。

        实测事故：一整轮 9 题评测共用同一个 ``services``，第一题烧穿预算后，
        其余 8 题全都在进门时被判超支——``actions == 0``、``evidence == 0``、
        直接拒答。用户看到的是"agentic 模式在这批题上全线失败"，
        而真实原因是一个跨题泄漏的累加器。
        """
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.max_steps = 3
        services = await factory.build()
        # 必须用"永不主动结束"的模型，否则循环一步就退出，
        # 而预算检查发生在**每轮开头**——根本不会被执行到。
        factory.llm_agent.complete = await looping_llm()  # type: ignore[method-assign]
        probe = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        single_session_tokens = probe.usage.total_tokens
        assert single_session_tokens > 0

        # 阈值刚好容得下一次会话，容不下两次
        factory.settings.agent.max_tokens = single_session_tokens + 1
        second = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert "budget_exceeded" not in second.notes, "第二题被第一题的花费挤掉了预算"
        assert second.usage.total_tokens <= single_session_tokens + 1

    async def test_usage_since_handles_an_untouched_baseline(self) -> None:
        """基线未动时增量就是自身；``cost_known`` 取两者的逻辑与。"""
        from scitrace.domain.session import Usage

        baseline = Usage(prompt_tokens=100, completion_tokens=50, cost_known=False)
        current = Usage(
            prompt_tokens=180, completion_tokens=90, estimated_cost=0.3,
            cost_currency="CNY", cost_known=True,
        )
        delta = current.since(baseline)
        assert delta.prompt_tokens == 80
        assert delta.completion_tokens == 40
        assert delta.estimated_cost == pytest.approx(0.3)
        assert delta.cost_currency == "CNY"
        assert delta.cost_known is False, "有一次读数不可信，增量成本就不可信"
        assert current.since(current).is_identity, "与自身之差必须是单位元"

    async def test_session_is_persisted(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        result = await AgentRuntime(services=services, question="Q").run()

        assert result.session_id
        stored = SessionStore(factory.settings.sessions_dir).load_sessions()
        assert result.session_id in stored
        assert stored[result.session_id].question == "Q"


class TestTerminationConditions:
    """SPEC §4.3 的每一种兜底都必须被触发过一次。"""

    async def test_max_steps_truncates(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.max_steps = 1
        services = await factory.build()
        factory.llm_agent.complete = await looping_llm()  # type: ignore[method-assign]
        result = await AgentRuntime(
            services=services, question="Q", mode="agentic"
        ).run()
        assert "max_steps_exceeded" in result.notes
        assert result.status is SessionStatus.TRUNCATED
        assert result.answer is not None

    async def test_wrap_up_does_not_discard_an_answer_the_model_already_gave(
        self, tmp_path: Path
    ) -> None:
        """**回归测试**：循环因兜底结束时，若模型已给出答案，不得重新合成。

        实测事故：一次 12 步会话在最后一步由模型自己答出 6,162 字符，
        撞上 ``max_steps`` 后强制收尾又花 39.96 秒重新合成一份 3,447 字符的答案，
        用户拿到的是模型**没有选择**的那一份。``answer_question`` 按设计
        可以不终止会话（供模型预览），所以循环结束时仍存在的答案
        就是模型最后认可的那一份，重做只会更差、更贵。

        模型必须**先收集证据再作答**：没有证据时 ``answer_question`` 会
        直接返回提示且不设置 ``state.answer``，于是循环末尾走的是
        "无证据即拒答"的早退分支，根本到不了被改的收尾逻辑。
        """
        from scitrace.ports import LLMResponse, ToolCall

        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.max_steps = 2
        services = await factory.build()
        calls = {"n": 0}

        async def scripted(messages, **kwargs):  # noqa: ANN001, ANN202, ARG001
            calls["n"] += 1
            tool = "gather_evidence" if calls["n"] == 1 else "answer_question"
            return LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id=f"c{calls['n']}",
                        name=tool,
                        arguments={"question": "Q"} if tool == "gather_evidence" else {},
                    ),
                ),
            )

        factory.llm_agent.complete = scripted  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()

        assert "max_steps_exceeded" in result.notes
        assert result.answer is not None
        assert not result.answer.refused
        answers = [a for a in result.actions if a.tool == "answer_question"]
        assert len(answers) == 1, "强制收尾重复合成了模型已经给出的答案"

    async def test_budget_truncates(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.max_cost = 0.0
        factory.settings.agent.max_tokens = 1
        services = await factory.build()
        factory.llm_agent.complete = await looping_llm()  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert "budget_exceeded" in result.notes
        assert result.status is SessionStatus.TRUNCATED

    async def test_timeout_truncates_but_still_answers(self, tmp_path: Path) -> None:
        """超时也**必须**拿得到答案：强制合成在超时作用域之外执行。

        写错这一点的典型症状是超时后返回空答案——用户等了 500 秒，
        拿到的是"没有结果"。
        """
        factory = FakeServicesFactory(tmp_path)
        # 超时必须发生在**已经拿到证据之后**，否则走的是"无证据拒答"分支，
        # 测不到"超时仍要交出答案"这条性质。
        factory.settings.agent.timeout_seconds = 0.05
        factory.settings.agent.max_steps = 10_000
        factory.settings.agent.no_new_evidence_limit = 10_000
        services = await factory.build()
        factory.llm_agent.complete = await looping_llm(delay=0.02)  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert "timeout" in result.notes
        assert result.status is SessionStatus.TRUNCATED
        assert result.answer.text, "超时后仍必须有答案正文"

    async def test_no_new_evidence_terminates(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.no_new_evidence_limit = 1
        factory.settings.agent.max_steps = 20
        services = await factory.build()
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        # 模型替身始终请求同一个工具，第二轮取证必然无新增
        assert result.answer is not None

    async def test_repeated_no_new_evidence_is_recorded(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.no_new_evidence_limit = 2
        factory.settings.agent.max_steps = 20
        services = await factory.build()

        async def always_gather(messages, **kwargs):  # noqa: ANN001
            from scitrace.ports import LLMResponse, ToolCall

            return LLMResponse(
                content="",
                tool_calls=(ToolCall(id="c1", name="gather_evidence", arguments={"question": "Q"}),),
                prompt_tokens=1,
                completion_tokens=1,
            )

        factory.llm_agent.complete = always_gather  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert "no_new_evidence" in result.notes

    async def test_malformed_tool_arguments_are_retried_then_fail(self, tmp_path: Path) -> None:
        """SPEC §4.1 的参数容错：回灌错误让模型重试，超过上限才终止。"""
        from scitrace.ports import LLMResponse, ToolCall

        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.max_tool_param_retries = 2
        services = await factory.build()

        async def always_bad(messages, **kwargs):  # noqa: ANN001
            return LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(
                        id="c1",
                        name="search_literature",
                        arguments={},
                        raw_arguments="{bad json",
                    ),
                ),
                prompt_tokens=1,
                completion_tokens=1,
            )

        factory.llm_agent.complete = always_bad  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert "tool_param_retries_exceeded" in result.notes

    async def test_unexpected_error_yields_fail_not_exception(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()

        async def boom(messages, **kwargs):  # noqa: ANN001
            raise RuntimeError("provider exploded")

        factory.llm_agent.complete = boom  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert result.status is SessionStatus.FAIL
        assert result.answer is not None, "失败也必须返回一个 Answer 对象"


class TestBehaviorContracts:
    """SPEC §8 的 C1–C8：把"逻辑一致"变成可执行的定义。"""

    async def test_c1_three_stage_order(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        result = await AgentRuntime(services=services, question="Q").run()
        tools = [action.tool for action in result.actions]
        assert {"search_literature", "gather_evidence", "answer_question"} <= set(tools)

    async def test_c2_successful_answer_always_has_citations(self, tmp_path: Path) -> None:
        services = await FakeServicesFactory(tmp_path).build()
        result = await AgentRuntime(services=services, question="Q").run()
        if result.status is SessionStatus.SUCCESS:
            assert result.answer.citations
            assert result.answer.references

    async def test_c3_refusal_has_no_citations(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path, answer_text=INSUFFICIENT_EVIDENCE)
        services = await factory.build()
        result = await AgentRuntime(services=services, question="Q").run()
        assert result.status in {SessionStatus.REFUSED, SessionStatus.UNSURE}
        assert result.answer.citations == []

    async def test_c4_low_relevance_evidence_never_enters_context(self, tmp_path: Path) -> None:
        """评分低于阈值的证据**绝不**出现在上下文中。"""
        factory = FakeServicesFactory(tmp_path)
        factory.screener.min_relevance = 10
        factory.screener.score = 9  # 低于阈值
        services = await factory.build()

        captured: list[str] = []
        original = factory.llm.complete

        async def spy(messages, **kwargs):  # noqa: ANN001
            captured.append(messages[-1].content)
            return await original(messages, **kwargs)

        factory.llm.complete = spy  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q").run()
        assert not result.answer.citations

    async def test_c5_index_rebuild_only_on_change(self, tmp_path: Path) -> None:
        """指纹只由表示配置决定——改检索策略不该导致索引重建。"""
        base = Settings()
        other = Settings()
        other.retrieval.strategy = "hybrid_rrf"
        assert base.index_fingerprint() == other.index_fingerprint()

        changed = Settings()
        changed.ingest.chunking.max_chars = changed.ingest.chunking.max_chars + 100
        assert base.index_fingerprint() != changed.index_fingerprint()

    async def test_c6_tool_semantics_are_stable(self, tmp_path: Path) -> None:
        """工具集的名字与 finish 的 has_answer 语义必须稳定。"""
        services = await FakeServicesFactory(tmp_path).build()
        names = {tool.name for tool in build_default_tools(services)}
        assert names == {
            "search_literature",
            "gather_evidence",
            "answer_question",
            "reset_scope",
            "finish",
        }

    async def test_c7_config_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """显式覆盖 > 进程环境 > 命名 profile > .env > 默认。"""
        from scitrace.config import load_settings
        from scitrace.config import settings as settings_module

        monkeypatch.setattr(settings_module, "CONFIG_ROOT", tmp_path / ".scitrace")
        monkeypatch.setenv("SCITRACE_LLM__MODEL", "from-env")
        assert load_settings(dotenv_path=None).llm.model == "from-env"
        assert (
            load_settings(dotenv_path=None, overrides={"llm": {"model": "from-cli"}}).llm.model
            == "from-cli"
        )

    async def test_c8_single_failure_does_not_break_the_run(self, tmp_path: Path) -> None:
        """单源失败不中断主流程：筛选器抛错也应产出结果而非异常。"""

        class ExplodingScreener(FakeScreenerWithEvidence):
            async def screen(self, question, fragments, *, sources):  # noqa: ANN001
                raise RuntimeError("metadata source down")

        factory = FakeServicesFactory(tmp_path)
        factory.screener = ExplodingScreener()
        services = await factory.build()
        result = await AgentRuntime(services=services, question="Q").run()
        assert isinstance(result.status, SessionStatus)
        assert result.answer is not None


class TestUsageFieldsAreCarriedThrough:
    """每次把 ``LLMResponse`` 折成 ``Usage`` 时都必须带上计价相关字段。

    这类字段漏传**不会报错**、不会让任何断言失败，只会静静地产生一个
    看起来正常的错数字——币种错标、缓存命中恒为 0。
    实测：agent 自身调用那一处漏了 ``cost_currency``，而它是 agentic 会话的
    第一次 merge，于是整个累加器被 ``Usage`` 的默认值 ``"USD"`` 污染，
    一次人民币计价的会话全程报 USD。
    """

    async def test_agent_call_currency_reaches_the_session(self, tmp_path: Path) -> None:
        from scitrace.ports import LLMResponse, ToolCall

        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()

        async def complete(messages, **kwargs):  # noqa: ANN001, ANN202, ARG001
            return LLMResponse(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="gather_evidence", arguments={"question": "Q"}),
                ),
                prompt_tokens=100,
                completion_tokens=20,
                cached_tokens=64,
                cost=0.5,
                cost_currency="CNY",
                cost_known=True,
            )

        factory.llm_agent.complete = complete  # type: ignore[method-assign]
        result = await AgentRuntime(services=services, question="Q", mode="agentic").run()
        assert result.usage.cost_currency == "CNY", "agent 调用的币种被丢掉了"
        assert result.usage.cached_tokens >= 64, "agent 调用的缓存命中数被丢掉了"

    def test_every_usage_from_a_response_carries_the_currency(self) -> None:
        """结构性护栏：所有从 LLM 响应构造 Usage 的地方都要带 cost_currency。"""
        import re

        root = Path(__file__).resolve().parents[1] / "src" / "scitrace"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for match in re.finditer(r"Usage\(\n(.*?)\n\s*\)", source, re.DOTALL):
                body = match.group(1)
                if "response.cost" not in body:
                    continue  # 不是从响应折出来的（如只带 dangling_citations）
                # 必须匹配**关键字参数**而非裸词：注释里提一句 cost_currency
                # 就足以骗过 `"cost_currency" in body`——第一版护栏正是这样失效的。
                if "cost_currency=" not in body:
                    line = source[: match.start()].count("\n") + 1
                    offenders.append(f"{path.relative_to(root)}:{line}")
        assert not offenders, f"这些 Usage 构造漏了 cost_currency：{offenders}"


class TestHistoryCompaction:
    """SPEC §4.2：历史超过 ``context_token_limit`` 时对旧观测做有损压缩。

    这个功能**只在长会话里触发**，日常测试跑不到，因此必须直接构造超限历史来测。
    它也是最容易写出"看起来对、上线就 400"的一类代码：压缩历史时若顺手删掉
    中间的消息，assistant 的 ``tool_calls`` 就会失去配对的 tool 消息。
    """

    def _runtime(self, tmp_path: Path, *, limit: int = 500):  # noqa: ANN202
        from scitrace.agent.runtime import AgentRuntime

        class _Usage:
            def __init__(self) -> None:
                from scitrace.domain.session import Usage

                self.usage = Usage()
                self.settings = type("S", (), {"pricing": type("P", (), {"currency": "CNY"})})()

        services = _Usage()
        runtime = AgentRuntime.__new__(AgentRuntime)
        runtime.settings = type(
            "S",
            (),
            {"agent": type("A", (), {"context_token_limit": limit})(), "sessions_dir": tmp_path},
        )()
        runtime.services = services
        from scitrace.agent.state import AgentState

        runtime.state = AgentState(question="Q")
        return runtime

    def _history(self, *, rounds: int, observation_chars: int = 600):  # noqa: ANN202
        from scitrace.ports import LLMMessage, ToolCall

        messages = [
            LLMMessage(role="system", content="你是研究助手。" + "规则" * 50),
            LLMMessage(role="user", content="问题：Q"),
        ]
        for index in range(rounds):
            call_id = f"c{index}"
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=f"第 {index} 轮决策",
                    tool_calls=(
                        ToolCall(id=call_id, name="gather_evidence", arguments={"question": "Q"}),
                    ),
                )
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    tool_call_id=call_id,
                    name="gather_evidence",
                    content=f"已收集证据 {index}\n" + "细节" * (observation_chars // 2),
                )
            )
        return messages

    def test_no_compaction_below_the_limit(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path, limit=10_000_000)
        messages = self._history(rounds=3)
        assert runtime._compact_history(messages) is messages, "未超限不该做任何拷贝"
        assert "history_compacted" not in runtime.state.notes

    def test_messages_are_never_dropped(self, tmp_path: Path) -> None:
        """**最重要的一条**：压缩只改 content，消息序列长度与角色必须原样保留。

        删掉中间的消息会让 assistant 的 ``tool_calls`` 失去配对的 tool 消息，
        多数提供商会直接返回 400——而那是在长会话里才发生的线上故障。
        """
        runtime = self._runtime(tmp_path, limit=500)
        messages = self._history(rounds=8)
        compacted = runtime._compact_history(messages)
        assert len(compacted) == len(messages)
        assert [m.role for m in compacted] == [m.role for m in messages]
        assert [m.tool_call_id for m in compacted] == [m.tool_call_id for m in messages]
        assert [m.tool_calls for m in compacted] == [m.tool_calls for m in messages]
        assert "history_compacted" in runtime.state.notes

    def test_head_and_tail_are_left_intact(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path, limit=500)
        messages = self._history(rounds=8)
        compacted = runtime._compact_history(messages)
        assert compacted[0].content == messages[0].content, "任务描述不能被压掉"
        assert compacted[1].content == messages[1].content
        for index in range(len(messages) - 4, len(messages)):
            assert compacted[index].content == messages[index].content, "正在处理的上下文不能被压掉"

    def test_old_observations_become_tool_name_plus_summary_line(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path, limit=500)
        messages = self._history(rounds=8)
        compacted = runtime._compact_history(messages)
        middle = compacted[2 : len(compacted) - 4]
        compressed = [m for m in middle if m.role == "tool" and "（已压缩）" in m.content]
        assert compressed, "中间应该有被压缩的旧观测"
        for message in compressed:
            assert message.content.startswith("gather_evidence（已压缩）：")
            assert "已收集证据" in message.content, "摘要行应保留观测首行的结论"
            assert "细节" not in message.content, "正文细节应被压掉"
            assert len(message.content) < 250

    def test_over_limit_after_compaction_is_reported_not_hidden(self, tmp_path: Path) -> None:
        """压缩完仍超限时必须如实标注，不能假装已经处理好了。"""
        runtime = self._runtime(tmp_path, limit=100)
        runtime._compact_history(self._history(rounds=8))
        assert "history_over_limit_after_compaction" in runtime.state.notes

    def test_observation_truncation_keeps_both_ends(self, tmp_path: Path) -> None:
        """截断保留首尾：结论常在末尾，只砍尾巴会把结论丢掉。"""
        from scitrace.agent.runtime import _OBSERVATION_CHAR_LIMIT

        runtime = self._runtime(tmp_path)
        text = "开头标记" + "中" * (_OBSERVATION_CHAR_LIMIT * 2) + "结尾标记"
        truncated = runtime._truncate_observation(text)
        assert truncated.startswith("开头标记")
        assert truncated.endswith("结尾标记")
        assert "省略" in truncated, "必须告知模型中间有内容被省略，静默截断会误导它"
        assert len(truncated) < len(text)

    def test_short_observation_is_untouched(self, tmp_path: Path) -> None:
        runtime = self._runtime(tmp_path)
        assert runtime._truncate_observation("已收集 3 条证据。") == "已收集 3 条证据。"


class _RecordingScreener:
    """记录每一次筛选调用收到了哪些片段，并可切换"是否放行"。"""

    def __init__(self, inner, *, accept: bool = True) -> None:  # noqa: ANN001
        self.inner = inner
        self.accept = accept
        self.calls: list[list[str]] = []
        self.last_usage = Usage(llm_calls=1, prompt_tokens=5, completion_tokens=5)

    async def screen(self, question, fragments, *, sources):  # noqa: ANN001
        self.calls.append([f.fragment_id for f in fragments])
        results = await self.inner.screen(question, fragments, sources=sources)
        return list(results) if self.accept else []


class TestScreenedFragmentDedup:
    """**已经筛过并被拒**的片段不得在后续轮次里重筛。

    同一片段对同一问题的评分不会改变，重筛一次就是白花一次 LLM 调用。
    这是本项目在"重复筛选"这一类缺陷上的第二处：第一处是"已入证片段被重筛"
    （已在 `GatherEvidenceTool` 预过滤里修掉），但预过滤只排除已入证的，
    被拒片段仍会每轮重筛。
    """

    async def test_rejected_fragments_are_not_screened_again(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        recorder = _RecordingScreener(services.screener)
        services.screener = recorder  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        state = AgentState(question="q")

        await tool.run({"question": "q"}, state)
        first = sum(len(c) for c in recorder.calls)
        assert first > 0, "第一轮应当有片段被送去筛选"

        await tool.run({"question": "q"}, state)
        second = sum(len(c) for c in recorder.calls) - first
        assert second == 0, f"同一批片段被重筛了 {second} 段（纯浪费）"

    async def test_screened_ids_are_recorded_even_when_rejected(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        recorder = _RecordingScreener(services.screener, accept=False)
        services.screener = recorder  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        state = AgentState(question="q")
        await tool.run({"question": "q"}, state)
        assert state.screened_fragment_ids, "被拒片段也必须记账，否则下一轮会重筛"
        assert state.evidence == []


class TestZeroEvidenceFallback:
    """主策略零证据时换一条检索通路再试一次（改进 C'）。

    实测依据：实体归属类问题在 `dense_mmr` 下会把正确文献排到第 3 名之后、
    片段相似度≈0，筛选全部拒绝 → 零证据 → 拒答（一道题上占 14%）。
    但**不能全局换策略**——全局 `hybrid_rrf` 会把跨论文综合打出 −30pp。
    """

    async def test_zero_evidence_triggers_a_fallback_retrieval(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.zero_evidence_fallback = True
        services = await factory.build()
        strategies: list[str | None] = []
        original = services.retriever.retrieve

        async def spy(query, **kwargs):  # noqa: ANN001, ANN202
            strategies.append(kwargs.get("strategy"))
            return await original(query, **kwargs)

        services.retriever.retrieve = spy  # type: ignore[method-assign]
        # 全部拒收 → added == 0 → 应当触发回退
        services.screener = _RecordingScreener(services.screener, accept=False)  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        await tool.run({"question": "q"}, AgentState(question="q"))
        assert strategies == [None, "hybrid_rrf"], f"回退未按预期触发：{strategies}"

    async def test_no_fallback_when_evidence_was_found(self, tmp_path: Path) -> None:
        """有证据时不得回退——回退是失败路径的补救，不是常态路径。"""
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        strategies: list[str | None] = []
        original = services.retriever.retrieve

        async def spy(query, **kwargs):  # noqa: ANN001, ANN202
            strategies.append(kwargs.get("strategy"))
            return await original(query, **kwargs)

        services.retriever.retrieve = spy  # type: ignore[method-assign]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        await tool.run({"question": "q"}, AgentState(question="q"))
        assert strategies == [None], "有证据却回退了，等于白白多跑一次检索"

    async def test_fallback_only_screens_fragments_the_main_path_missed(
        self, tmp_path: Path
    ) -> None:
        """回退**绝不能**重筛主路径已筛过的片段。

        同一片段对同一问题评分不变，重筛是纯浪费——而回退恰恰是最该省的地方，
        因为它只在已经出问题的时候触发。
        """
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.zero_evidence_fallback = True
        services = await factory.build()
        recorder = _RecordingScreener(services.screener, accept=False)
        services.screener = recorder  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        state = AgentState(question="q")
        await tool.run({"question": "q"}, state)

        all_ids = [fid for call in recorder.calls for fid in call]
        assert len(all_ids) == len(set(all_ids)), (
            f"有片段被重复筛选：{len(all_ids)} 次调用 vs {len(set(all_ids))} 个唯一片段"
        )


class TestZeroEvidenceFallbackIsOffByDefault:
    """回退**默认关闭**——它是质量与 token 之间的取舍，不是免费改进。

    实测（EXPERIMENTS.md 实验二十二）：打开后 TRUNCATED 9→4、SUCCESS 14→18，
    但每题 tokens +11%（P90 +22%），且因证据集变大触发了一次
    `synthesis_failed`（推理 token 吃光预算）。默认必须是省 token 的那一侧。
    """

    async def test_no_fallback_by_default(self, tmp_path: Path) -> None:
        factory = FakeServicesFactory(tmp_path)
        services = await factory.build()
        strategies: list[str | None] = []
        original = services.retriever.retrieve

        async def spy(query, **kwargs):  # noqa: ANN001, ANN202
            strategies.append(kwargs.get("strategy"))
            return await original(query, **kwargs)

        services.retriever.retrieve = spy  # type: ignore[method-assign]
        services.screener = _RecordingScreener(services.screener, accept=False)  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "gather_evidence")
        await tool.run({"question": "q"}, AgentState(question="q"))
        assert strategies == [None], "默认配置下不该发生回退"


class TestEvidenceBudgetForSynthesis:
    """送进合成的证据必须封顶。

    实测依据（EXPERIMENTS.md 实验二十二）：证据集大小是尾部成本与合成失败率的
    **共同驱动**——一次收集了 16 条证据的运行触发了合成返回空输出的失败，
    而该题平时只有 1–4 条。

    **刻意只封顶"送进合成的部分"，不驱逐证据集**：驱逐会让已展示给模型的
    引用键消失，可能产生悬空引用，而 SPEC §1.3 要求引用可回溯率为 100%。
    """

    def test_selection_keeps_the_most_relevant(self) -> None:
        from scitrace.agent.tools import _select_for_synthesis

        class _Item:
            def __init__(self, key: str, relevance: int) -> None:
                self.key, self.relevance = key, relevance

        items = [_Item(f"e{i}", score) for i, score in enumerate([3, 9, 5, 8, 1])]
        picked = _select_for_synthesis(items, 3)
        assert [item.key for item in picked] == ["e1", "e3", "e2"]

    def test_selection_is_stable_for_equal_scores(self) -> None:
        """同分保持原序——排序不稳定会让同一问题两次运行看到不同材料。"""
        from scitrace.agent.tools import _select_for_synthesis

        class _Item:
            def __init__(self, key: str) -> None:
                self.key, self.relevance = key, 5

        items = [_Item(f"e{i}") for i in range(6)]
        assert [i.key for i in _select_for_synthesis(items, 3)] == ["e0", "e1", "e2"]

    def test_no_budget_keeps_everything(self) -> None:
        """默认不限——实测设上限会让覆盖率掉 24pp（实验二十四）。"""
        from scitrace.agent.tools import _select_for_synthesis

        class _Item:
            def __init__(self) -> None:
                self.relevance = 5

        items = [_Item() for _ in range(40)]
        assert len(_select_for_synthesis(items, None)) == 40

    def test_under_budget_keeps_everything(self) -> None:
        from scitrace.agent.tools import _select_for_synthesis

        class _Item:
            def __init__(self) -> None:
                self.relevance = 5

        items = [_Item() for _ in range(4)]
        assert len(_select_for_synthesis(items, 12)) == 4

    async def test_synthesis_receives_at_most_the_budget(self, tmp_path: Path) -> None:
        """端到端：证据远超预算时，合成看到的条数必须被截到预算。"""
        factory = FakeServicesFactory(tmp_path)
        factory.settings.agent.evidence_budget = 2
        services = await factory.build()
        seen: list[int] = []

        class _Spy:
            last_usage = Usage(llm_calls=1, prompt_tokens=5, completion_tokens=5)

            async def synthesize(self, question, evidence, *, sources):  # noqa: ANN001, ARG002
                seen.append(len(evidence))
                from scitrace.domain import Answer

                return Answer(text="答 (c)", raw_text="答", citations=[])

        services.synthesizer = _Spy()  # type: ignore[assignment]
        tool = next(t for t in build_default_tools(services) if t.name == "answer_question")
        state = AgentState(question="q")
        state.evidence = [
            Evidence(
                key=f"ev-{i:08x}",
                source_key="src-1",
                fragment_id=f"f{i}",
                summary="s",
                relevance=5 + (i % 3),
                citation="(c)",
                page_label="1",
            )
            for i in range(9)
        ]
        await tool.run({}, state)
        assert seen == [2], f"合成收到了 {seen} 条证据，预算未生效"
        assert len(state.evidence) == 9, "证据集本身不该被驱逐（会产生悬空引用风险）"
        assert any(n.startswith("evidence_trimmed") for n in state.notes)

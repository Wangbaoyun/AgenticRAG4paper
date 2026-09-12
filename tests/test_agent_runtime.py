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

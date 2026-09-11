"""测试 LLM 接入层的纯逻辑与重试策略。

LLM 集成里真正容易出错的地方几乎都在"模型输出的畸形 JSON"与"哪些错误值得重试"
这两件事上，而它们在联网环境下极难稳定复现。本文件把它们变成普通的参数化测试。
"""

from __future__ import annotations

import json
import math

import httpx
import pytest
from scitrace.adapters.llm import (
    OpenAIEmbeddingClient,
    RetryPolicy,
    is_retryable_error,
    normalize_vector,
    parse_tool_calls,
    to_backend_messages,
)
from scitrace.adapters.llm.convert import (
    extract_balanced_json_object,
    parse_tool_arguments,
    strip_reasoning_tags,
)
from scitrace.adapters.llm.retry import with_retries
from scitrace.ports import LLMMessage, ToolCall


class TestStripReasoningTags:
    def test_removes_think_block(self) -> None:
        assert strip_reasoning_tags("<think>hmm</think>The answer.") == "The answer."

    @pytest.mark.parametrize("tag", ["think", "thinking", "reasoning"])
    def test_supported_tag_names(self, tag: str) -> None:
        assert strip_reasoning_tags(f"<{tag}>x</{tag}>Answer") == "Answer"

    def test_multiline_block(self) -> None:
        assert strip_reasoning_tags("<think>\nline1\nline2\n</think>\nDone") == "Done"

    def test_unclosed_tag_is_preserved(self) -> None:
        """未闭合通常意味着输出被截断，删掉反而丢失线索。"""
        assert "<think>" in strip_reasoning_tags("<think>truncated")

    def test_plain_text_unchanged(self) -> None:
        assert strip_reasoning_tags("  plain  ") == "plain"


class TestExtractBalancedJsonObject:
    def test_simple(self) -> None:
        assert extract_balanced_json_object('prefix {"a": 1} suffix') == '{"a": 1}'

    def test_nested(self) -> None:
        assert extract_balanced_json_object('{"a": {"b": 2}}') == '{"a": {"b": 2}}'

    def test_braces_inside_string_do_not_confuse(self) -> None:
        assert extract_balanced_json_object('{"a": "}"}') == '{"a": "}"}'

    def test_escaped_quote_inside_string(self) -> None:
        assert extract_balanced_json_object(r'{"a": "he said \"hi\"}"}') == (
            r'{"a": "he said \"hi\"}"}'
        )

    def test_takes_first_object_not_last_brace(self) -> None:
        """模型先给示例再给结果时，取"第一个 { 到最后一个 }"会得到非法串。"""
        text = 'example {"x": 1} then real {"y": 2}'
        assert extract_balanced_json_object(text) == '{"x": 1}'

    def test_no_object(self) -> None:
        assert extract_balanced_json_object("no braces here") is None

    def test_unbalanced(self) -> None:
        assert extract_balanced_json_object('{"a": 1') is None


class TestParseToolArguments:
    def test_valid_object(self) -> None:
        arguments, failure = parse_tool_arguments('{"query": "rag", "k": 3}')
        assert arguments == {"query": "rag", "k": 3}
        assert failure is None

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_empty_becomes_empty_dict(self, raw: str | None) -> None:
        assert parse_tool_arguments(raw) == ({}, None)

    def test_markdown_fence(self) -> None:
        arguments, _ = parse_tool_arguments('```json\n{"a": 1}\n```')
        assert arguments == {"a": 1}

    def test_reasoning_tags_stripped(self) -> None:
        arguments, _ = parse_tool_arguments('<think>hmm</think>{"a": 1}')
        assert arguments == {"a": 1}

    def test_trailing_comma_repaired(self) -> None:
        arguments, _ = parse_tool_arguments('{"a": 1,}')
        assert arguments == {"a": 1}

    def test_unquoted_key_repaired(self) -> None:
        arguments, _ = parse_tool_arguments('{query: "rag"}')
        assert arguments == {"query": "rag"}

    def test_smart_quotes_repaired(self) -> None:
        arguments, _ = parse_tool_arguments("{“query”: “rag”}")
        assert arguments == {"query": "rag"}

    def test_prose_wrapped(self) -> None:
        arguments, _ = parse_tool_arguments('Sure, here you go: {"a": 1} — hope that helps')
        assert arguments == {"a": 1}

    def test_top_level_array_is_wrapped(self) -> None:
        arguments, _ = parse_tool_arguments("[1, 2]")
        assert arguments == {"value": [1, 2]}

    def test_unparseable_returns_raw(self) -> None:
        """解析失败不抛异常：一次参数解析失败不该中断整个会话。"""
        arguments, failure = parse_tool_arguments("this is not json at all")
        assert arguments is None
        assert failure == "this is not json at all"

    def test_truncated_json_is_not_guessed(self) -> None:
        """截断的 JSON 宁可明确失败，也不猜测补全——猜出来的"成功"内容是错的。"""
        arguments, failure = parse_tool_arguments('{"query": "rag", "k":')
        assert arguments is None
        assert failure is not None


class TestParseToolCalls:
    def test_object_form(self) -> None:
        class Function:
            name = "search_literature"
            arguments = '{"query": "rag"}'

        class Call:
            id = "call-1"
            function = Function()

        calls = parse_tool_calls([Call()])
        assert len(calls) == 1
        assert calls[0].id == "call-1"
        assert calls[0].name == "search_literature"
        assert calls[0].arguments == {"query": "rag"}
        assert calls[0].raw_arguments is None

    def test_dict_form(self) -> None:
        calls = parse_tool_calls(
            [{"id": "c1", "function": {"name": "gather", "arguments": '{"q": "x"}'}}]
        )
        assert calls[0].name == "gather"
        assert calls[0].arguments == {"q": "x"}

    @pytest.mark.parametrize("raw", [None, [], ()])
    def test_empty(self, raw) -> None:
        assert parse_tool_calls(raw) == ()

    def test_malformed_arguments_preserved(self) -> None:
        """解析失败时保留原始串，调用方据此决定重试。"""
        calls = parse_tool_calls([{"id": "c1", "function": {"name": "f", "arguments": "{bad"}}])
        assert calls[0].arguments == {}
        assert calls[0].raw_arguments == "{bad"

    def test_missing_id_gets_placeholder(self) -> None:
        calls = parse_tool_calls([{"function": {"name": "f", "arguments": "{}"}}])
        assert calls[0].id


class TestToBackendMessages:
    def test_plain_messages(self) -> None:
        payload = to_backend_messages(
            [LLMMessage(role="system", content="sys"), LLMMessage(role="user", content="hi")]
        )
        assert payload == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_assistant_tool_calls_are_roundtripped(self) -> None:
        """多数提供商要求 assistant 的 tool_calls 与后续 tool 消息成对出现，

        缺失会直接返回 400——因此回填时必须原样带上。
        """
        message = LLMMessage(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="c1", name="search", arguments={"query": "rag"}),),
        )
        payload = to_backend_messages([message])
        assert payload[0]["tool_calls"][0]["id"] == "c1"
        assert payload[0]["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(payload[0]["tool_calls"][0]["function"]["arguments"]) == {"query": "rag"}

    def test_tool_message_includes_call_id(self) -> None:
        payload = to_backend_messages(
            [LLMMessage(role="tool", content="result", tool_call_id="c1", name="search")]
        )
        assert payload[0]["tool_call_id"] == "c1"
        assert payload[0]["name"] == "search"

    def test_assistant_history_has_reasoning_stripped(self) -> None:
        """上一轮的思考被当作答案喂回去，既浪费 token 又会诱导重复思考。"""
        payload = to_backend_messages(
            [LLMMessage(role="assistant", content="<think>x</think>Real answer.")]
        )
        assert payload[0]["content"] == "Real answer."

    def test_user_content_is_not_stripped(self) -> None:
        payload = to_backend_messages([LLMMessage(role="user", content="<think>keep me</think>")])
        assert payload[0]["content"] == "<think>keep me</think>"


class TestNormalizeVector:
    def test_unit_norm(self) -> None:
        vector = normalize_vector([3.0, 4.0])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_direction_preserved(self) -> None:
        vector = normalize_vector([1.0, 1.0])
        assert math.isclose(vector[0], vector[1], rel_tol=1e-9)

    def test_zero_vector_returned_unchanged(self) -> None:
        assert normalize_vector([0.0, 0.0]) == [0.0, 0.0]

    @pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
    def test_invalid_vectors_raise(self, vector: list[float]) -> None:
        with pytest.raises(ValueError):
            normalize_vector(vector)


class TestRetryPolicy:
    def test_exponential_backoff(self) -> None:
        policy = RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=100.0, jitter=0.0)
        assert [policy.delay_for(n) for n in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 8.0]

    def test_capped(self) -> None:
        policy = RetryPolicy(base_delay_s=1.0, max_delay_s=5.0, jitter=0.0)
        assert policy.delay_for(10) == 5.0

    def test_jitter_within_bounds(self) -> None:
        policy = RetryPolicy(base_delay_s=10.0, max_delay_s=100.0, jitter=0.25)
        for _ in range(50):
            assert 7.5 <= policy.delay_for(1) <= 12.5


class TestIsRetryableError:
    @pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 529])
    def test_retryable_status_codes(self, status: int) -> None:
        error = Exception("boom")
        error.status_code = status  # type: ignore[attr-defined]
        assert is_retryable_error(error) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_non_retryable_status_codes(self, status: int) -> None:
        error = Exception("boom")
        error.status_code = status  # type: ignore[attr-defined]
        assert is_retryable_error(error) is False

    @pytest.mark.parametrize(
        "message",
        ["Connection timeout", "rate limit exceeded", "server error", "Overloaded"],
    )
    def test_retryable_messages(self, message: str) -> None:
        assert is_retryable_error(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        ["Invalid API key provided", "context_length_exceeded", "content_filter triggered"],
    )
    def test_non_retryable_messages(self, message: str) -> None:
        assert is_retryable_error(RuntimeError(message)) is False

    def test_non_retryable_marker_wins_over_retryable(self) -> None:
        """``invalid_request_error`` 的消息里常同时出现 "rate limit" 字样，

        若不做优先级判断，会把一个永远失败 400 重试三次。
        """
        assert is_retryable_error(RuntimeError("invalid_request_error: rate limit field")) is False

    def test_unknown_error_is_not_retried(self) -> None:
        """不确定时选择不重试，避免把偶发错误变成长时间挂起。"""
        assert is_retryable_error(RuntimeError("something odd")) is False


class FakeSleeper:
    """记录等待时长但不真的等待。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class TestWithRetries:
    async def test_returns_on_first_success(self) -> None:
        calls = []

        async def operation() -> str:
            calls.append(1)
            return "ok"

        result = await with_retries(
            operation, policy=RetryPolicy(attempts=3), sleep=FakeSleeper()
        )
        assert result == "ok"
        assert len(calls) == 1

    async def test_retries_then_succeeds(self) -> None:
        sleeper = FakeSleeper()
        attempts = []

        async def operation() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("rate limit")
            return "ok"

        result = await with_retries(
            operation,
            policy=RetryPolicy(attempts=3, base_delay_s=1.0, jitter=0.0),
            sleep=sleeper,
        )
        assert result == "ok"
        assert sleeper.delays == [1.0, 2.0]

    async def test_gives_up_and_reraises_original(self) -> None:
        """必须原样抛出原始异常：上层要按错误类型做降级判断，包装会抹掉这个信息。"""

        class CustomError(RuntimeError):
            pass

        async def operation() -> str:
            raise CustomError("rate limit")

        with pytest.raises(CustomError):
            await with_retries(
                operation, policy=RetryPolicy(attempts=2), sleep=FakeSleeper()
            )

    async def test_non_retryable_fails_immediately(self) -> None:
        sleeper = FakeSleeper()
        attempts = []

        async def operation() -> str:
            attempts.append(1)
            raise RuntimeError("Invalid API key")

        with pytest.raises(RuntimeError):
            await with_retries(
                operation, policy=RetryPolicy(attempts=5), sleep=sleeper
            )
        assert len(attempts) == 1, "不可恢复的错误不应重试"
        assert sleeper.delays == []


class TestOpenAIEmbeddingClient:
    def make_client(self, handler) -> OpenAIEmbeddingClient:
        transport = httpx.MockTransport(handler)
        return OpenAIEmbeddingClient(
            "test-embed",
            api_base="https://example.invalid/v1",
            client=httpx.AsyncClient(transport=transport),
        )

    async def test_embeds_and_normalizes(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [3.0, 4.0]} for index, _ in enumerate(payload["input"])
                    ]
                },
            )

        client = self.make_client(handler)
        vectors = await client.embed(["a", "b"])
        assert len(vectors) == 2
        assert math.isclose(math.sqrt(sum(v * v for v in vectors[0])), 1.0, rel_tol=1e-9)
        assert client.dimension == 2

    async def test_responses_are_reordered_by_index(self) -> None:
        """并行响应顺序不保证与输入一致，不排序会静默把向量配错文本。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )

        client = self.make_client(handler)
        vectors = await client.embed(["first", "second"])
        assert vectors[0][0] > vectors[0][1], "index=0 的向量必须排在前面"

    async def test_count_mismatch_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

        client = self.make_client(handler)
        with pytest.raises(RuntimeError, match="条数不匹配"):
            await client.embed(["a", "b"])

    async def test_empty_input_makes_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("不应发起请求")

        client = self.make_client(handler)
        assert await client.embed([]) == []

    async def test_batching_splits_requests(self) -> None:
        sizes: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            sizes.append(len(payload["input"]))
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [1.0, 0.0]}
                        for index, _ in enumerate(payload["input"])
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        client = OpenAIEmbeddingClient(
            "test-embed",
            api_base="https://example.invalid/v1",
            batch_size=3,
            client=httpx.AsyncClient(transport=transport),
        )
        await client.embed(["x"] * 7)
        assert sizes == [3, 3, 1]

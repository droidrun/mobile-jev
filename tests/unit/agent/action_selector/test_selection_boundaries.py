import asyncio
import json
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, TextBlock, ToolCallBlock
from llama_index.core.llms import MockLLM
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration
from openinference.instrumentation import TracerProvider
from pydantic import BaseModel

from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm import parsing
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse, achat
from mobilerun_agents.agent.logs.memory import MemoryDetail
from mobilerun_agents.agent.tools.base import tool
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import (
    AgentConfig,
    DecisionConfig,
    MobileConfig,
    ToolsConfig,
)
from tests.unit.agent.action_selector.fixtures import Turn


async def typed(value: int, enabled: bool, size: float, mode: Literal["a", "b"]) -> None: ...


async def text(value: str) -> None: ...


async def nested(values: list[dict[str, Literal["a", "b"]]] | None) -> None: ...


def test_nested_argument_enums_and_nullable_arrays():
    action = Action.compile("test", "nested", nested, Declaration())
    assert parsing.coerce_arguments(action.input_schema, {"values": '[{"x":"b"}]'}) == {
        "values": [{"x": "b"}]
    }
    assert parsing.coerce_arguments(action.input_schema, {"values": "null"}) == {"values": None}
    with pytest.raises(InvalidResponse):
        parsing.coerce_arguments(action.input_schema, {"values": '[{"x":"other"}]'})


def test_argument_types_belong_to_each_action():
    action = Action.compile("test", "typed", typed, Declaration())
    raw = dict(value="42", enabled="false", size="2.5", mode="b")
    assert parsing.coerce_arguments(action.input_schema, raw) == dict(
        value=42, enabled=False, size=2.5, mode="b"
    )
    string = Action.compile("test", "text", text, Declaration())
    assert parsing.coerce_arguments(string.input_schema, {"value": "42"}) == {"value": "42"}


@pytest.mark.parametrize(
    "key,bad",
    [
        ("value", "3.2"),
        ("value", True),
        ("size", "NaN"),
        ("size", "1e999"),
        ("enabled", "yes"),
        ("enabled", 1),
        ("enabled", "TRUE"),
        ("mode", "private-note"),
    ],
)
def test_malformed_argument_reports_only_safe_feedback(key, bad):
    action = Action.compile("test", "typed", typed, Declaration())
    raw = dict(value=1, enabled=True, size=1, mode="a")
    raw[key] = bad
    with pytest.raises(InvalidResponse, match="^Invalid arguments$"):
        parsing.coerce_arguments(action.input_schema, raw)


@pytest.mark.parametrize("protocol", ["xml", "native"])
def test_reasoning_and_tool_arguments_stay_separate(protocol):
    notes = "<add_memory> one </add_memory>\n<add_memory>two</add_memory>"
    if protocol == "xml":
        blocks = [
            TextBlock(
                text=notes
                + '<function_calls><invoke name="type"><parameter name="text"><add_memory>payload</add_memory></parameter></invoke></function_calls>'
            )
        ]
    else:
        blocks = [
            TextBlock(text=notes),
            ToolCallBlock(
                tool_name="type", tool_kwargs={"text": "<add_memory>payload</add_memory>"}
            ),
        ]
    llm = AsyncMock()
    llm.metadata = MockLLM().metadata
    llm.achat.return_value = ChatResponse(message=ChatMessage(role="assistant", blocks=blocks))

    @tool
    async def type(text: str) -> None:
        """Type the requested text."""

    turn = Turn(goal="Type")
    turn.registry = build_registry(turn.device, [type], ToolsConfig())
    result = asyncio.run(
        turn.select(LlamaIndexLLMSelector(llm, supports_native=protocol == "native"))
    )
    assert result.thought == notes
    assert result.calls[0].tool_kwargs["text"] == "<add_memory>payload</add_memory>"


def test_configured_timeout_cancels_stream_consumption():
    async def chunks():
        yield ChatResponse(message=ChatMessage(content="Started"))
        await asyncio.Event().wait()

    llm = AsyncMock()
    llm.metadata = MockLLM().metadata
    llm.astream_chat.return_value = chunks()
    selector = LlamaIndexLLMSelector(llm, config=AgentConfig(decision=DecisionConfig(timeout=0.02)))
    with pytest.raises(TimeoutError, match="LLM request timed out"):
        asyncio.run(asyncio.wait_for(Turn(goal="Wait").select(selector, stream=True), 1))


@pytest.mark.parametrize("log_response", [False, True])
def test_stream_consumption_respects_tracing_privacy(log_response):
    tracer = TracerProvider().get_tracer(__name__)

    async def chunks():
        with tracer.start_as_current_span("consume provider response") as span:
            assert span.is_recording() is log_response
            yield ChatResponse(message=ChatMessage(content="Done"))

    async def run():
        llm = AsyncMock()
        llm.astream_chat.return_value = chunks()
        reply = await achat(llm, [], stream=True, log_response=log_response)
        assert reply.message.content == "Done"
        with tracer.start_as_current_span("after provider response") as span:
            assert span.is_recording()

    asyncio.run(run())


class RequiredOutput(BaseModel):
    count: int


def _llm(*calls, native=True):
    if native:
        message = ChatMessage(
            role="assistant",
            blocks=[ToolCallBlock(tool_name=name, tool_kwargs=args) for name, args in calls],
        )
    else:
        invokes = []
        for name, args in calls:
            params = "".join(
                f'<parameter name="{key}">{value if type(value) is str else json.dumps(value)}</parameter>'
                for key, value in args.items()
            )
            invokes.append(f'<invoke name="{name}">{params}</invoke>')
        message = ChatMessage(
            role="assistant", content=f"<function_calls>{''.join(invokes)}</function_calls>"
        )
    return AsyncMock(
        metadata=MockLLM().metadata, achat=AsyncMock(return_value=ChatResponse(message=message))
    )


def _select(turn, selector):
    return asyncio.run(selector.select(turn.ctx, turn.device, turn.memory, turn.registry))


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("output", [{"count": "not an integer"}, None])
def test_typed_completion_defers_output_validation(native, output):
    turn = Turn()
    turn.registry = build_registry(turn.device, [], ToolsConfig(), output_model=RequiredOutput)
    arguments = {"success": True, "message": "Found it", "output": output}
    llm = _llm(("complete", arguments), native=native)
    selector = LlamaIndexLLMSelector(
        llm, supports_native=native, config=AgentConfig(streaming=False)
    )

    result = _select(turn, selector)

    assert result.calls[0].tool_name == "complete"
    assert result.calls[0].tool_kwargs == arguments
    assert turn.registry.validate(result.calls[0]) is None
    if not native:
        prompt = llm.achat.call_args.args[0][0].content
        assert "output must match this JSON schema" in prompt
        assert "RequiredOutput" in prompt


@pytest.mark.parametrize(
    "native,invalid",
    [
        (True, {"unexpected": 1}),
        (False, {"unexpected": 1}),
        (True, {"success": 1}),
        (False, {"success": "yes"}),
        (True, {"message": 9}),
        (True, {"output": [1]}),
        (False, {"output": [1]}),
    ],
)
def test_typed_completion_rejects_bad_dispatch_arguments(native, invalid):
    turn = Turn()
    turn.registry = build_registry(turn.device, [], ToolsConfig(), output_model=RequiredOutput)
    arguments = {"success": True, "message": "Found it", "output": {"count": 1}, **invalid}
    selector = LlamaIndexLLMSelector(
        _llm(("complete", arguments), native=native),
        supports_native=native,
        config=AgentConfig(streaming=False),
    )

    with pytest.raises(InvalidResponse, match="^Invalid arguments$"):
        _select(turn, selector)


def test_hand_built_selector_resolves_config_at_select_time(tmp_path):
    prompt = tmp_path / "test.jinja2"
    prompt.write_text("Custom prompt. Parallel: {{ parallel_tools }}")
    llm = _llm(("wait", {}), ("wait", {}))
    with ConfigLoader.override(MobileConfig(agent=AgentConfig(streaming=False))):
        selector = LlamaIndexLLMSelector(llm, supports_native=True)

    config = AgentConfig(system_prompt=str(prompt), streaming=False, parallel_tools=False)
    with ConfigLoader.override(MobileConfig(agent=config)):
        with pytest.raises(InvalidResponse):
            _select(Turn(), selector)

    assert llm.achat.call_args.args[0][0].content == "Custom prompt. Parallel: False"
    assert llm.achat.call_args.kwargs["allow_parallel_tool_calls"] is False


def test_explicit_config_takes_priority_over_active_override():
    llm = _llm(("wait", {}), ("wait", {}))
    selector = LlamaIndexLLMSelector(
        llm, supports_native=True, config=AgentConfig(streaming=False, parallel_tools=True)
    )

    with ConfigLoader.override(MobileConfig(agent=AgentConfig(parallel_tools=False))):
        result = _select(Turn(), selector)

    assert [call.tool_name for call in result.calls] == ["wait", "wait"]


def test_llm_memory_detail_identifies_the_selector(caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    turn = Turn()
    turn.memory.memory_blocks = []
    turn.memory._estimate_token_count = lambda _: 0
    turn.memory.token_limit = 100
    selector = LlamaIndexLLMSelector(
        _llm(("wait", {})), supports_native=True, config=AgentConfig(streaming=False)
    )

    _select(turn, selector)

    details = [event.selector for event in turn.events if type(event) is MemoryDetail]
    assert details == ["llm"]

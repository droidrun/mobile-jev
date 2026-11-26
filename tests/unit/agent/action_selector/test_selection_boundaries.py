import asyncio
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, TextBlock, ToolCallBlock
from llama_index.core.llms import MockLLM
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration
from openinference.instrumentation import TracerProvider

from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm import parsing
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse, achat
from mobilerun_agents.config.user_config import DecisionConfig
from tests.unit.agent.selector_fixtures import selection_run


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
def test_only_reasoning_notes_become_memory(protocol):
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
    result = asyncio.run(
        LlamaIndexLLMSelector(llm, supports_native=protocol == "native").select(
            selection_run(goal="Type")
        )
    )
    assert result.memory_update == "one\ntwo"
    assert result.actions[0].parameters["text"] == "<add_memory>payload</add_memory>"


def test_configured_timeout_cancels_stream_consumption():
    async def chunks():
        yield ChatResponse(message=ChatMessage(content="Started"))
        await asyncio.Event().wait()

    llm = AsyncMock()
    llm.metadata = MockLLM().metadata
    llm.astream_chat.return_value = chunks()
    selector = LlamaIndexLLMSelector(llm, config=DecisionConfig(timeout=0.02))
    with pytest.raises(TimeoutError, match="LLM request timed out"):
        asyncio.run(asyncio.wait_for(selector.select(selection_run(goal="Wait", stream=True)), 1))


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

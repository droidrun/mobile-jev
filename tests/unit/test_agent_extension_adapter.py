"""Generic device extension tools and public context."""

import asyncio
import logging
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ToolCallBlock
from mobilerun.core.extensions import Extension, action
from mobilerun.models.context import ContextItem

from mobilerun.agents.tools.actions import click
from mobilerun.agents.tools.signatures import build_tool_registry
from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.fast_agent.fast_agent import FastAgent
from tests.unit.device_fixtures import MemoryDevice


class Choice(str, Enum):
    ONE = "one"


class Example(Extension):
    default_name = "example"

    @action(sensitive_parameters=frozenset({"secret"}))
    async def send(self, secret: str, choice: Choice) -> dict[str, str]:
        """Send a private value."""
        return {"choice": choice.value}

    @action()
    async def status(self) -> str:
        """Read status."""
        return "ready"

    @action()
    async def context_argument(self, ctx: str) -> str:
        """Return the argument named ctx."""
        return ctx

    async def context(self):
        return (ContextItem(self.name, "status", {"state": "ready"}),)


def test_extension_tools_schema_execution_redaction_and_context():
    async def run():
        device = MemoryDevice()
        await Example().bind(device)
        registry, _ = await build_tool_registry(device=device)
        spec = next(s for s in registry.get_function_tool_specs() if s["name"] == "example__send")
        assert spec["parameters"] == device.actions.get("example.send").input_schema
        assert "example__status" in registry.tools
        result_with_ctx = await registry.execute(
            "example__context_argument",
            {"ctx": "allowed"},
            SimpleNamespace(driver=device),
        )
        assert result_with_ctx.success and '"allowed"' in result_with_ctx.summary
        assert "type_secret" not in registry.tools

        class Events:
            def __init__(self):
                self.events = []

            def write_event_to_stream(self, event):
                self.events.append(event)

        events = Events()

        class Ctx:
            driver = device

        result = await registry.execute(
            "example__send",
            {"secret": "sensitive-value", "choice": "one"},
            Ctx(),
            events,
        )
        assert result.success and '"one"' in result.summary
        assert "sensitive-value" not in str(events.events)
        assert "sensitive-value" not in registry.redact_text(
            "example__send",
            {"secret": "sensitive-value", "choice": "one"},
            "response contains sensitive-value",
        )
        await device.close()

    asyncio.run(run())


def test_index_click_dispatches_through_action_catalog():
    ui = SimpleNamespace(
        get_element_coords=lambda index: (12, 34),
        get_element_info=lambda index: {},
        use_normalized=False,
    )
    actions = SimpleNamespace(call=AsyncMock())
    ctx = SimpleNamespace(
        driver=SimpleNamespace(actions=actions),
        ui=ui,
        macro_recorder=None,
        state_provider=SimpleNamespace(requires_coordinate_tools=False),
    )
    result = asyncio.run(click(2, ctx=ctx))
    assert result.success
    actions.call.assert_awaited_once_with("device.tap", {"x": 12, "y": 34}, audience="operator")


def test_fast_agent_history_and_events_redact_sensitive_argument(caplog, tmp_path):
    from mobilerun.core.devices.recording import RecordingDevice
    from mobilerun.core.state.ui_state import UIState
    from mobilerun.macro.recorder import MacroRecorder
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from test_fast_agent_native_tools import _PromptResolver, _StateProvider

    from mobilerun.agents.trajectory.trajectory import Trajectory
    from mobilerun.agents.trajectory.writer import TrajectoryWriter

    caplog.set_level(logging.DEBUG)

    async def run():
        device = RecordingDevice(MemoryDevice())
        await Example().bind(device)
        registry, _ = await build_tool_registry(device=device)
        state = MobileAgentState()
        macro_recorder = MacroRecorder()
        replies = iter(
            [
                ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        blocks=[
                            ToolCallBlock(
                                tool_call_id="1",
                                tool_name="example__send",
                                tool_kwargs={
                                    "secret": "canary-secret-value",
                                    "choice": "one",
                                },
                            )
                        ],
                    )
                ),
                ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        blocks=[
                            ToolCallBlock(
                                tool_call_id="2",
                                tool_name="complete",
                                tool_kwargs={"success": True, "message": "done"},
                            )
                        ],
                    )
                ),
            ]
        )

        async def respond(*args, **kwargs):
            return next(replies)

        class StateProvider(_StateProvider):
            async def get_state(self):
                return UIState(elements=[], formatted_text="", focused_text="", phone_state={})

        agent = FastAgent(
            llm=SimpleNamespace(supports_native_tool_calls=True, native_tool_api="responses"),
            agent_config=SimpleNamespace(
                fast_agent=SimpleNamespace(vision=False, parallel_tools=False, tool_calling="auto"),
                max_steps=10,
                streaming=False,
                after_sleep_action=0,
            ),
            registry=registry,
            action_ctx=SimpleNamespace(
                driver=device,
                ui=None,
                shared_state=state,
                macro_recorder=macro_recorder,
            ),
            state_provider=StateProvider(),
            shared_state=state,
            prompt_resolver=_PromptResolver(),
        )
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        instrumentor = LlamaIndexInstrumentor()
        instrumentor.instrument(tracer_provider=provider)
        with (
            patch(
                "mobilerun.agents.workflows.fast_agent.fast_agent.acall_with_retries",
                new=respond,
            ),
            patch(
                "mobilerun.agents.workflows.fast_agent.fast_agent.get_usage_from_response",
                return_value=None,
            ),
        ):
            handler = agent.run(input="test")
            events = [event async for event in handler.stream_events()]
            await handler
        spans = exporter.get_finished_spans()
        assert spans
        assert "canary-secret-value" not in str([span.attributes for span in spans])
        instrumentor.uninstrument()
        provider.shutdown()
        assert "canary-secret-value" not in str(events)
        assert "canary-secret-value" not in str(state.message_history)
        assert "canary-secret-value" not in caplog.text
        assert "canary-secret-value" not in str(device.recorded_actions)
        assert macro_recorder.actions and "canary-secret-value" not in str(macro_recorder.actions)
        assert any(item["action_type"] == "extension_call" for item in device.recorded_actions)
        trajectory = Trajectory(base_path=str(tmp_path))
        trajectory.events.extend(events)
        trajectory.macro = macro_recorder.actions
        writer = TrajectoryWriter()
        await writer.start()
        writer.write(trajectory, "test")
        await writer.stop()
        for path in trajectory.trajectory_folder.rglob("*"):
            if path.is_file():
                assert b"canary-secret-value" not in path.read_bytes()
        await device.close()

    asyncio.run(run())


def test_executor_mocked_turn_presents_schema_and_executes_action(caplog):
    from mobilerun.agents.config.user_config import AgentConfig
    from mobilerun.agents.workflows.executor.executor_agent import ExecutorAgent

    caplog.set_level(logging.DEBUG)

    async def run():
        device = MemoryDevice()
        await Example().bind(device)
        registry, _ = await build_tool_registry(device=device)
        state = MobileAgentState()
        config = AgentConfig()
        config.executor.vision = False
        config.after_sleep_action = 0
        seen = []

        async def respond(llm, messages, **kwargs):
            seen.extend(messages)
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content='### Thought\nUse canary-secret-value\n### Action\n{"action":"example__send", "secret":"canary-secret-value", "choice":"one"}\n### Description\nUse canary-secret-value',
                )
            )

        agent = ExecutorAgent(
            llm=SimpleNamespace(),
            registry=registry,
            action_ctx=SimpleNamespace(driver=device),
            shared_state=state,
            agent_config=config,
        )
        with (
            patch(
                "mobilerun.agents.workflows.executor.executor_agent.acall_with_retries",
                new=respond,
            ),
            patch(
                "mobilerun.agents.workflows.executor.executor_agent.get_usage_from_response",
                return_value=None,
            ),
        ):
            handler = agent.run(subgoal="read status")
            events = [str(event) async for event in handler.stream_events()]
            result = await handler
        assert result["outcome"] is True
        assert '"$defs"' in str(seen) and "example__send" in str(seen)
        assert "canary-secret-value" not in str(events)
        assert "canary-secret-value" not in caplog.text
        await device.close()

    asyncio.run(run())


def test_stateless_manager_mocked_turn_does_not_request_extension_context():
    from test_fast_agent_native_tools import _StateProvider

    from mobilerun.agents.config.user_config import AgentConfig
    from mobilerun.agents.workflows.manager.stateless_manager_agent import (
        StatelessManagerAgent,
    )

    async def run():
        class NoContextDevice(MemoryDevice):
            async def context(self):
                raise AssertionError("agent must not request extension context")

        device = NoContextDevice()
        await Example().bind(device)
        config = AgentConfig()
        config.manager.vision = False
        state = MobileAgentState(instruction="open settings")
        seen = []

        async def respond(llm, messages, **kwargs):
            seen.extend(messages)
            return ChatResponse(
                message=ChatMessage(role="assistant", content="<plan>1. Open settings</plan>")
            )

        manager = StatelessManagerAgent(
            llm=SimpleNamespace(),
            action_ctx=SimpleNamespace(driver=device, ui=None),
            state_provider=_StateProvider(),
            shared_state=state,
            agent_config=config,
        )
        with (
            patch(
                "mobilerun.agents.workflows.manager.stateless_manager_agent.acall_with_retries",
                new=respond,
            ),
            patch(
                "mobilerun.agents.workflows.manager.stateless_manager_agent.get_usage_from_response",
                return_value=None,
            ),
        ):
            result = await manager.run()
        assert result["plan"]
        assert "open settings" in str(seen)
        await device.close()

    asyncio.run(run())


def test_sensitive_llm_turn_suppresses_provider_span_capture(caplog):
    from llama_index.core.llms.mock import MockLLM
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from mobilerun.agents.llm.inference import acall_with_retries

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        message = ChatMessage(role="user", content="canary-secret-value")
        response = asyncio.run(acall_with_retries(MockLLM(), [message], log_response=False))
        assert "canary-secret-value" in str(response)
        assert "canary-secret-value" not in str(exporter.get_finished_spans())
        assert "canary-secret-value" not in caplog.text
    finally:
        instrumentor.uninstrument()
        provider.shutdown()

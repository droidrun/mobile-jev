import asyncio
import json
import socket
from datetime import date
from enum import Enum
from inspect import signature
from typing import Literal
from unittest.mock import AsyncMock, Mock, NonCallableMock

import pytest
from llama_index.core.workflow import Context
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel
from workflows.errors import WorkflowCancelledByUser

from mobilerun_agents.agent.action_selector.base import ActionSelection, ActionSelector
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.workflows.agent import AgentWorkflow
from mobilerun_agents.agent.workflows.events import Act, Invalid, Select, StepFinished, UserMessage
from tests.unit.agent.selector_fixtures import captured_turn
from tests.unit.device_fixtures import MemoryDevice


class Output(BaseModel):
    count: int


class Dated(BaseModel):
    day: date


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("workflow unit tests cannot open network sockets")

    for method in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, method, blocked)


class Phone(MemoryDevice):
    async def tap(self, x: int, y: int) -> None:
        await self.tapped(x, y)

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        return b"screen"


class Selector(ActionSelector):
    def __init__(self, *calls):
        self.calls = iter(calls)
        self.requests = []

    async def select(self, run):
        self.requests.append(await captured_turn(run))
        return ActionSelection(actions=[next(self.calls)])


def complete(output=None):
    return ToolCall(
        "complete",
        {"success": True, "message": "done", **({"output": output} if output is not None else {})},
    )


async def resources(output=None):
    device = Phone(supported_operations=frozenset({"tap"}))
    provider = NonCallableMock(spec=StateProvider)
    provider.requires_coordinate_tools = False
    provider.capabilities = {"element_index"}
    provider.get_state = AsyncMock(
        return_value=UIState([{"index": 3, "bounds": "0,0,20,40"}], "screen", "", {}, 100, 100)
    )
    device.tapped = AsyncMock()
    device._set_state_provider(provider)
    return dict(device=device, output=output, goal="count items")


@pytest.mark.anyio
async def test_public_actions_hide_sensitive_arguments_without_changing_dispatch():
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.tools.base import complete as finish_tool

    seen = []
    private = "fake-private-action-value"

    @tool(sensitive_parameters=frozenset({"value"}))
    async def sensitive(run: Run, value: str) -> str:
        """Use a sensitive value."""
        seen.append(value)
        return "ok"

    class Echoing(Selector):
        async def select(self, run):
            selection = await super().select(run)
            selection.thought = selection.response_text = private
            return selection

    calls = [ToolCall("sensitive", {"value": private}), complete()]
    handler = AgentWorkflow(Echoing(*calls)).run(
        **await resources(), tools=[sensitive, finish_tool]
    )
    acts = [event async for event in handler.stream_events() if type(event) is Act]
    assert (await handler).success and seen == [private]
    assert len(acts) == 2
    assert acts[0].selection.actions[0].parameters == {"value": "[REDACTED]"}
    assert all(private not in repr(event.selection) for event in acts)


@pytest.mark.anyio
@pytest.mark.parametrize("encode", [lambda v: json.dumps(v)[1:-1], lambda v: repr(v)[1:-1], str])
async def test_act_response_text_masks_encoded_private_arguments(encode):
    from mobilerun_agents import Run, tool

    private = "fake 'quote' \"fixture\"\n"

    @tool(sensitive_parameters=frozenset({"value"}))
    async def enter(run: Run, value: str) -> None:
        """Use a sensitive value."""

    class Echo(Selector):
        async def select(self, run):
            selection = await super().select(run)
            selection.response_text = f"argument: {encode(private)}"
            return selection

    handler = AgentWorkflow(Echo(ToolCall("enter", {"value": private}))).run(
        **await resources(), tools=[enter], max_steps=1
    )
    acts = [event async for event in handler.stream_events() if type(event) is Act]
    await handler
    assert acts[0].selection.response_text == "argument: ***"


@pytest.mark.anyio
async def test_malformed_call_does_not_echo_private_arguments_in_stream():
    from mobilerun_agents import Run, tool

    private = 'fake-"invalid"-marker'

    @tool(sensitive_parameters=frozenset({"value"}))
    async def enter(run: Run, value: str) -> None:
        """Use a sensitive value."""

    bad = ToolCall("enter", {"value": private}, error=f"Invalid value: {json.dumps(private)}")
    handler = AgentWorkflow(Selector(bad, complete())).run(**await resources(), tools=[enter])
    errors = [event.error async for event in handler.stream_events() if type(event) is Invalid]
    assert (await handler).success
    assert errors == ['Invalid value: "***"']


@pytest.mark.anyio
@pytest.mark.parametrize("record_first", [False, True])
async def test_recorder_primitive_cannot_replay_a_tracked_secret(tmp_path, record_first):
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.operators._consumers import RunConsumers
    from mobilerun_agents.config.user_config import MobileConfig

    private = 'fake-"primitive"-marker'

    @tool(sensitive_parameters=frozenset({"value"}))
    async def enter(run: Run, value: str) -> None:
        """Use a sensitive value."""

    @tool
    async def record(run: Run) -> None:
        """Record one replayable primitive."""
        assert run.macro_recorder is not None
        run.macro_recorder.record_action(
            {"action_type": "open_url", "url": f"https://example.invalid/{private}"}
        )

    config = MobileConfig()
    config.analytics.enabled = config.tracing.enabled = False
    config.checkpoints.enabled = True
    config.checkpoints.folder = tmp_path / "runs"
    consumers = RunConsumers(config, "privacy-test", 3)
    assert consumers.recorder is not None
    consumers.recorder.macro_path = tmp_path / "macro.json"
    calls = [ToolCall("enter", {"value": private}), ToolCall("record")]
    if record_first:
        calls.reverse()
    flow = AgentWorkflow(Selector(*calls, complete()))
    async with consumers:
        handler = flow.run(**await resources(), tools=[enter, record])
        events = [event async for event in handler.stream_events()]
        for event in events:
            await consumers.on_event(event)
        state = await handler.ctx.store.get_state()
        await consumers.finish(await handler, state)
    streamed = next(
        e.action for e in events if type(e) is StepFinished and e.action.name == "open_url"
    )
    macro = json.loads(consumers.recorder.macro_path.read_text())
    assert macro["actions"] == []
    if not record_first:
        assert streamed.args == {"url": "https://example.invalid/***"}
        assert not streamed.replayable
        assert next(a for a in state.actions if a.name == "open_url") == streamed


@pytest.mark.anyio
async def test_normal_completion_does_not_echo_a_prior_sensitive_argument(caplog):
    from mobilerun_agents import AgentOperator, Run, tool
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    private = "fake-private-completion-value"

    @tool(sensitive_parameters=frozenset({"value"}))
    async def sensitive(run: Run, value: str) -> str:
        """Use a sensitive value."""
        return "ok"

    config = MobileConfig()
    config.analytics.enabled = config.checkpoints.enabled = config.tracing.enabled = False
    args = await resources()
    calls = [
        ToolCall("sensitive", {"value": private}),
        ToolCall("complete", {"success": True, "message": f"Done with {private}"}),
    ]
    with ConfigLoader.override(config), caplog.at_level("INFO", logger="mobilerun_agents"):
        result = await AgentOperator(Selector(*calls), [sensitive]).execute(
            args["device"], args["goal"]
        )
    assert result.success and result.answer == "Done with ***"
    assert "Done with ***" in caplog.text and private not in caplog.text


@pytest.mark.anyio
async def test_reused_operator_loads_each_execute_override():
    from mobilerun_agents import AgentOperator, Run, tool
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    @tool
    async def progress(run: Run) -> str:
        """Take a fake step without completing the task."""
        return "ok"

    operator = AgentOperator(Selector(*[ToolCall("progress")] * 3), [progress])
    args = await resources()
    for budget in (1, 2):
        config = MobileConfig()
        config.analytics.enabled = config.checkpoints.enabled = config.tracing.enabled = False
        config.agent.max_steps = budget
        with ConfigLoader.override(config):
            result = await operator.execute(args["device"], args["goal"])
        assert not result.success and result.steps == budget


@pytest.mark.anyio
async def test_full_run_stream_and_checkpoint():
    selector = Selector(ToolCall("tap_element", {"index": 3}), complete({"count": 2}))
    flow = AgentWorkflow(selector)
    args = await resources(Output)
    provider = args["device"].state._resolved_provider
    shown = provider.get_state.return_value
    moved = UIState([{"index": 3, "bounds": "70,80,90,100"}], "moved", "", {}, 100, 100)
    provider.get_state.side_effect = [shown, moved, moved]
    handler = flow.run(**args)
    events = [e async for e in handler.stream_events()]
    result = await handler
    assert result.success and result.output == Output(count=2) and result.steps == 2
    assert [type(e) for e in events if isinstance(e, (Select, StepFinished))] == [
        Select,
        StepFinished,
        Select,
        StepFinished,
    ]
    args["device"].tapped.assert_awaited_once_with(10, 20)
    assert next(e for e in events if isinstance(e, Select)).screenshot is None
    first = next(e.action for e in events if isinstance(e, StepFinished))
    assert first.name == "tap" and first.args == {"x": 10, "y": 20}
    restored = Context.from_dict(flow, handler.ctx.to_dict())
    assert (await restored.store.get_state()).actions[-1].name == "complete"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "calls,limit,reason",
    [
        ([complete({}), complete({"count": 4})], 3, "complete"),
        ([complete({})] * 2, 4, "output"),
        ([ToolCall("delegate")] * 3, 1, "repair"),
        ([ToolCall("tap_element", {"index": 3})], 1, "max_steps"),
    ],
)
async def test_budgets(calls, limit, reason):
    selector = Selector(*calls)
    result = await AgentWorkflow(selector).run(**await resources(Output), max_steps=limit)
    assert reason in result.reason
    assert result.success == (reason == "complete")
    assert len(selector.requests) == len(calls)


@pytest.mark.anyio
@pytest.mark.parametrize("valid", [False, True])
async def test_strict_json_completion(valid):
    from mobilerun_agents.config.user_config import AgentConfig

    payload = {"day": "2026-11-24"} if valid else {"count": "2"}
    flow = AgentWorkflow(Selector(complete(payload), complete(payload)))
    config = AgentConfig(max_output_validation_attempts=2)
    handler = flow.run(**await resources(Dated if valid else Output), config=config)
    if valid:
        result = await handler
        assert result.success and result.output == Dated(day=date(2026, 11, 24))
    else:
        result = await handler
        assert not result.success and result.reason == "output_attempts" and result.steps == 2


@pytest.mark.anyio
@pytest.mark.parametrize("error", [None, RuntimeError("input rejected")], ids=["false", "raises"])
async def test_native_input_failure_does_not_export_private_text(error):
    from mobilerun_agents.agent.checkpoints.macro_export import macro_data

    args = await resources()
    device = args["device"]
    device._supported_operations = frozenset({"tap", "input_text"})
    device._input_text_raw = AsyncMock(return_value=False, side_effect=error)
    selector = Selector(ToolCall("input_text", {"text": "hello"}), complete())
    handler = AgentWorkflow(selector).run(**args)
    events = [e async for e in handler.stream_events() if isinstance(e, StepFinished)]
    actions = [e.action for e in events]
    assert (await handler).success
    assert [(a.name, a.ok) for a in actions[:-1]] == [("input_text", False)]
    assert actions[0].args["text"] == "[REDACTED]"
    assert not any("ok" in a.args for a in actions)
    macro = macro_data(args["goal"], ((e.action, e.pre_state) for e in events))
    assert macro["actions"] == []


@pytest.mark.anyio
async def test_no_dispatch_after_cancel():
    entered = asyncio.Event()

    class Blocked(Selector):
        async def select(self, run):
            entered.set()
            await asyncio.Event().wait()
            return await super().select(run)

    args = await resources()
    handler = AgentWorkflow(Blocked(ToolCall("tap_element", {"index": 3}))).run(**args)
    await asyncio.wait_for(entered.wait(), 5)
    await handler.cancel_run()
    with pytest.raises(WorkflowCancelledByUser):
        await handler
    args["device"].tapped.assert_not_awaited()


@pytest.mark.anyio
async def test_external_entry_point(monkeypatch):
    from types import ModuleType

    from mobilerun_agents.agent.workflows.external import ExternalWorkflow

    plugin = ModuleType("sample")

    async def run(device, instruction, config, max_steps, output_schema):
        assert instruction == "count items" and output_schema["properties"]["count"]
        return dict(success=True, reason="complete", answer="ok", output={"count": 2}, steps=1)

    plugin.run = run
    entry = Mock()
    entry.load.return_value = plugin
    monkeypatch.setattr(
        "mobilerun_agents.agent.workflows.external.metadata.entry_points",
        lambda *, group, name: [entry] if (group, name) == ("mobilerun.agents", "sample") else [],
    )
    args = await resources(Output)
    result = await ExternalWorkflow("sample").run(
        device=args["device"], goal="count items", output=Output
    )
    assert result.output == Output(count=2)


def test_external_workflow_does_not_accept_an_action_selector():
    from mobilerun_agents.agent.workflows.external import ExternalWorkflow

    with pytest.raises(TypeError, match="action_selector"):
        ExternalWorkflow("sample", action_selector=Selector())


@pytest.mark.anyio
async def test_user_message_is_visible_next_turn():
    entered, resume = asyncio.Event(), asyncio.Event()

    class Paused(Selector):
        async def select(self, run):
            if not self.requests:
                entered.set()
                await resume.wait()
            return await super().select(run)

    selector = Paused(ToolCall("tap_element", {"index": 3}), complete())
    handler = AgentWorkflow(selector).run(**await resources())
    await asyncio.wait_for(entered.wait(), 5)
    handler.ctx.send_event(UserMessage(text="Use the second row"))
    resume.set()
    assert (await handler).success
    assert "Use the second row" in selector.requests[-1].memory


@pytest.mark.anyio
async def test_default_selector_factory_and_concurrent_runs(monkeypatch):
    from mobilerun_agents.agent.workflows import resources as workflow_resources

    factory_mock = AsyncMock(side_effect=lambda config: Selector(complete()))
    monkeypatch.setattr(workflow_resources, "default_selector", factory_mock)
    flow = AgentWorkflow()
    a, b = await resources(), await resources()
    b["goal"] = "other run"
    first, second = flow.run(**a), flow.run(**b)
    results = await asyncio.gather(first, second)
    assert all(r.success and r.steps == 1 for r in results)
    assert (await first.ctx.store.get_state()).goal == "count items"
    assert (await second.ctx.store.get_state()).goal == "other run"
    assert factory_mock.await_count == 2


@pytest.mark.anyio
async def test_run_resource_is_per_invocation_and_state_is_detached():
    from mobilerun_agents import Run, tool

    seen = {}

    @tool
    async def inspect_run(run: Run) -> str:
        """Verify the resource used for dispatch."""
        assert run is seen[run.state.goal]
        return "ok"

    class Inspect(ActionSelector):
        async def select(self, run):
            state = run.state
            assert not state.notes
            assert seen.setdefault(state.goal, run) is run
            state.notes.append("not persisted")
            state.goal = "changed locally"
            return ActionSelection(actions=[complete() if state.step else ToolCall("inspect_run")])

    from mobilerun_agents.agent.tools.base import complete as finish_tool

    flow = AgentWorkflow(Inspect())
    args = await resources()
    first = flow.run(**args, tools=[inspect_run, finish_tool])
    second = flow.run(**{**args, "goal": "second"}, tools=[inspect_run, finish_tool])
    assert all(result.success for result in await asyncio.gather(first, second))
    assert seen["count items"] is not seen["second"]
    for handler in (first, second):
        checkpoint = json.dumps(handler.ctx.to_dict())
        assert "not persisted" not in checkpoint and "agent.run" not in checkpoint
        restored = Context.from_dict(flow, json.loads(checkpoint))
        assert (await restored.store.get_state()).goal in seen


@pytest.mark.anyio
async def test_only_transient_selection_errors_retry(monkeypatch):
    from workflows.plugins.basic import InternalAsyncioAdapter
    from workflows.runtime.types import step_function

    # The worker stamps failures; the adapter schedules wakeups against that clock.
    now = 1000.0
    attempts_at = []
    wait = InternalAsyncioAdapter.wait_for_next_task

    async def get_now(adapter):
        return now

    async def advance_on_wait(adapter, running, pending, timeout=None):
        nonlocal now
        if timeout is not None:
            now += timeout
            timeout = 0
        return await wait(adapter, running, pending, timeout)

    monkeypatch.setattr(step_function, "time", Mock(time=lambda: now))
    monkeypatch.setattr(InternalAsyncioAdapter, "get_now", get_now)
    monkeypatch.setattr(InternalAsyncioAdapter, "wait_for_next_task", advance_on_wait)

    class Unavailable(Selector):
        async def select(self, run):
            attempts_at.append(now)
            if len(attempts_at) == 1:
                raise ConnectionError("unavailable")
            return await super().select(run)

    selector = Unavailable(complete())
    result = await AgentWorkflow(selector, timeout=None).run(**await resources())
    assert result.success and result.steps == 1
    assert attempts_at == [1000.0, 1005.0]


@pytest.mark.anyio
async def test_factory_selector_uses_native_registry_and_typed_completion(monkeypatch):
    from llama_index.core.base.llms.types import ToolCallBlock
    from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM

    from mobilerun_agents.agent.action_selector import loader
    from mobilerun_agents.config.user_config import LLMProfile, MobileConfig

    class Chat(MockLLM):
        async def achat(self, messages, **kw):
            assert "complete" in [s.metadata.get_name() for s in kw["tools"]]
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(
                            tool_name="complete",
                            tool_kwargs={"success": True, "message": "ok", "output": {"count": 7}},
                        )
                    ],
                )
            )

    config = MobileConfig(
        llm_profiles={"default": LLMProfile(provider="openai", model="fake", vision="off")}
    )
    config.agent.streaming = False
    monkeypatch.setattr(loader, "build_llm", AsyncMock(return_value=Chat()))
    selector = await loader.build_selector(config)
    result = await AgentWorkflow(selector).run(**await resources(Output), config=config.agent)
    assert result.success and result.output == Output(count=7)


@pytest.mark.anyio
async def test_jev_uses_agent_credentials_and_indexes_the_observed_ui(monkeypatch):
    from mobilerun_agents.agent.action_selector import DecisionSelector
    from mobilerun_agents.agent.action_selector.decision_model import typesafe

    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")
    client = AsyncMock()

    async def select_tap(**kwargs):
        criteria = kwargs["questions"]["action"].criteria
        choice = next(
            k for k, v in criteria.items() if type(v) is dict and v["tool"] == "tap_element"
        )
        return {"choices": {"action": {"choice": choice, "confidence": 1}}}

    client.system_one.side_effect = select_tap
    scope = AsyncMock()
    scope.__aenter__.return_value = client
    factory = Mock(return_value=scope)
    monkeypatch.setattr(typesafe, "AsyncTypeSafeClient", factory)
    args = await resources()
    args["device"].state._resolved_provider.get_state.return_value.elements[0].update(
        clickable=True, text="Count", type="button"
    )
    flow = AgentWorkflow(DecisionSelector(typesafe.Jev(), "off"))
    handler = flow.run(**args, max_steps=1)
    events = [event async for event in handler.stream_events()]
    result = await handler
    picked = next(event.selection for event in events if isinstance(event, Act))
    assert picked.actions == [ToolCall("tap_element", {"index": 3})]
    assert picked.description == 'tap_element [3] "Count" (button)'
    assert "candidate" not in signature(ActionSelection).parameters
    assert "candidate" not in signature(flow._dispatch).parameters
    assert result.reason == "max_steps"
    args["device"].tapped.assert_awaited_once_with(10, 20)
    factory.assert_called_once_with(api_key="fake-test-key")


@pytest.mark.anyio
async def test_jev_runs_parameterless_and_literal_extension_actions():
    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.action_selector import DecisionSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision

    calls = []

    class Scale(str, Enum):
        celsius = "celsius"

    class Bench(Extension):
        default_name = "bench"

        @action
        async def ping(self) -> None:
            """Check bench status."""
            calls.append("ping")

        @action
        async def defaults(self, check: bool = True) -> None:
            """Use defaults for the bench check."""

        @action
        async def choose(self, mode: Literal["warm", "cold"], enabled: bool, scale: Scale) -> None:
            """Choose bench mode."""
            calls.append((mode, enabled, scale is Scale.celsius))

        @action
        async def free(self, text: str) -> None:
            """Set arbitrary text."""

        @action(sensitive_parameters=frozenset({"key"}))
        async def secret(self, key: Literal["fake-a", "fake-b"]) -> None:
            """Use a sensitive key."""

    class Model:
        max_options = None
        locates_points = False

        async def decide(self, request):
            options = [
                (i, c) for i, c in enumerate(request.options) if c.action.name.startswith("bench__")
            ]
            assert [c.action for _, c in options] == [
                ToolCall("bench__choose", {"enabled": enabled, "mode": mode, "scale": "celsius"})
                for enabled in (False, True)
                for mode in ("cold", "warm")
            ] + [ToolCall("bench__defaults", {"check": check}) for check in (False, True)] + [
                ToolCall("bench__ping")
            ]
            pick = options[-1] if not calls else options[0]
            return Decision(option_index=pick[0])

    args = await resources()
    ext = await Bench().bind(args["device"])
    try:
        result = await AgentWorkflow(DecisionSelector(Model(), "off")).run(**args, max_steps=2)
    finally:
        await ext.unbind()
    assert result.reason == "max_steps"
    assert calls == ["ping", ("cold", False, True)]


@pytest.mark.anyio
async def test_notes_round_trip_in_workflow_state():
    class Notes(Selector):
        async def select(self, run):
            choice = await super().select(run)
            if len(self.requests) == 1:
                choice.memory_update = "Remember row seven"
            return choice

    selector = Notes(ToolCall("tap_element", {"index": 3}), complete())
    flow = AgentWorkflow(selector)
    handler = flow.run(**await resources(), max_steps=7)
    assert (await handler).success
    assert "Remember row seven" in selector.requests[1].memory
    restored = Context.from_dict(flow, handler.ctx.to_dict())
    state = await restored.store.get_state()
    assert state.notes == ["Remember row seven"] and state.max_steps == 7

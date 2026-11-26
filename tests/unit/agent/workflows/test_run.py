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


@pytest.mark.anyio
async def test_unchanged_wait_forces_delegation_despite_high_confidence():
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision
    from mobilerun_agents.config.user_config import ToolsConfig
    from tests.unit.agent.action_selector.test_selectors import FakeModel

    class WaitModel(FakeModel):
        calls = 0
        max_options = None

        async def decide(self, request):
            self.calls += 1
            return Decision(
                option_index=next(
                    i for i, option in enumerate(request.options) if option.action.name == "wait"
                ),
                confidence=0.9,
            )

    primary = WaitModel()
    delegate = Selector(complete())
    flow = AgentWorkflow(DelegatingSelector(DecisionSelector(primary, vision="off"), delegate, 0.7))
    result = await flow.run(**await resources(), tool_config=ToolsConfig(wait_duration=0.001))
    assert result.success and result.steps == 3 and primary.calls == 2
    [request] = delegate.requests
    assert (
        "wait" not in request.by_name
        and "wait repeated 2 times, screen unchanged" in request.memory
    )


@pytest.mark.anyio
async def test_stuck_wait_fails_after_four_executions_and_is_replanned():
    from mobilerun_agents.agent.planner import Finish, Plan
    from mobilerun_agents.agent.workflows.planning import PlanningWorkflow
    from mobilerun_agents.config.user_config import ToolsConfig
    from tests.unit.agent.workflows.test_planning import ScriptedPlanner

    wait = ToolCall("wait")
    planner = ScriptedPlanner(Plan(("recover",)), Finish(False, "blocked"))
    handler = PlanningWorkflow(planner, Selector(wait, wait, wait, wait, wait, complete())).run(
        **await resources(), tool_config=ToolsConfig(wait_duration=0.001)
    )
    events = [e async for e in handler.stream_events()]
    outcome = await handler
    assert outcome.steps == 4 and not outcome.success
    [subgoal] = planner.turns[-1].results
    assert not subgoal.success and subgoal.message == "stuck: wait ×4, screen unchanged"
    assert any(isinstance(event, Invalid) and "screen unchanged" in event.error for event in events)


@pytest.mark.anyio
async def test_llm_guard_hides_wait_and_repairs_a_repeat():
    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
    from mobilerun_agents.agent.logs.diagnostics import private_values
    from mobilerun_agents.agent.workflows.events import GuardDetail
    from mobilerun_agents.config.user_config import ToolsConfig
    from tests.unit.agent.workflows.test_planning import PlannerLLM

    llm = PlannerLLM(*[("wait", {})] * 3, ("complete", complete().parameters))
    selector = LlamaIndexLLMSelector(llm, vision="off", native_tool_api="responses")
    token = private_values.set(set())
    try:
        handler = AgentWorkflow(selector).run(
            **await resources(), tool_config=ToolsConfig(wait_duration=0.001)
        )
        events = [event async for event in handler.stream_events()]
        result = await handler
    finally:
        private_values.reset(token)
    assert result.success and result.steps == 3
    messages, tools = llm._turns[2]
    assert "wait" not in {tool.metadata.get_name() for tool in tools}
    assert any(
        "wait repeated 2 times, screen unchanged; try something else" in (msg.content or "")
        for msg in messages
    )
    assert sum(isinstance(event, Invalid) for event in events) == 1
    assert [event.section for event in events if isinstance(event, GuardDetail)] == ["guard"]


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["screen", "arguments", "action"])
async def test_progress_does_not_trigger_stuck_guard(change):
    from mobilerun_agents.config.user_config import ToolsConfig

    args = await resources()
    phone = args["device"]
    phone._supported_operations = frozenset({"open_url"})
    provider = phone.state._resolved_provider
    opened = []

    async def open_url(url: str) -> None:
        opened.append(url)
        if change == "screen":
            provider.get_state.return_value = UIState(
                [{"index": 3, "text": str(len(opened))}], "screen", "", {}, 100, 100
            )

    phone.open_url = open_url
    calls = [
        (
            ToolCall("wait")
            if change == "action" and i % 2
            else ToolCall(
                "open_url", {"url": f"https://example.invalid/{i if change == 'arguments' else 0}"}
            )
        )
        for i in range(4)
    ]
    handler = AgentWorkflow(Selector(*calls, complete())).run(
        **args, tool_config=ToolsConfig(wait_duration=0.001)
    )
    events = [event async for event in handler.stream_events()]
    assert (await handler).success and len(opened) == (2 if change == "action" else 4)
    assert not any(isinstance(event, Invalid) for event in events)


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
async def test_failed_plain_text_action_is_visible_but_not_replayed(error):
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
    assert actions[0].args["text"] == "hello"
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
async def test_setup_default_and_concurrent_runs(monkeypatch):
    from mobilerun_agents.agent.action_selector.loader import SetupSpec
    from mobilerun_agents.agent.workflows import resources as workflow_resources

    factory_mock = AsyncMock(side_effect=lambda: SetupSpec(selector=Selector(complete())))
    monkeypatch.setattr(workflow_resources, "default_setup", factory_mock)
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
        profiles={"default": LLMProfile(provider="openai", model="fake", vision="off")}
    )
    config.agent.streaming = False
    monkeypatch.setattr(loader, "build_llm", AsyncMock(return_value=Chat()))
    from mobilerun_agents.config.loader import ConfigLoader

    with ConfigLoader.override(config):
        selector = (await loader.default_setup()).selector
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


@pytest.mark.anyio
@pytest.mark.parametrize("reason", ["complete", "max_steps"])
async def test_private_notes_stay_raw_for_model_and_callback_but_not_in_context(reason):
    from mobilerun_agents import Run, tool

    secret = "FAKE password / test=84927"
    reported = []

    @tool(sensitive_parameters=frozenset({"token"}))
    async def confidential(run: Run, token: str) -> None:
        """Use a fake private token."""

    class Notes(Selector):
        async def select(self, run):
            choice = await super().select(run)
            if len(self.requests) == 1:
                choice.memory_update = f"raw {secret}"
            else:
                assert run.state.notes == [f"raw {secret}"]
                choice.memory_update = f"later {secret}"
            return choice

    private = ToolCall("confidential", {"token": secret})
    selector = Notes(private, complete() if reason == "complete" else private)
    handler = AgentWorkflow(selector).run(
        **await resources(), tools=[confidential], max_steps=2, report_notes=reported.append
    )
    events = [event async for event in handler.stream_events()]
    assert (await handler).reason == reason
    assert reported == [(f"raw {secret}", f"later {secret}")]
    assert f"raw {secret}" in selector.requests[1].memory
    assert secret not in repr(events)
    assert secret not in json.dumps(handler.ctx.to_dict())
    assert (await handler.ctx.store.get_state()).notes == ["raw ***", "later ***"]


@pytest.mark.anyio
async def test_inflight_sensitive_tool_argument_is_not_serialized():
    from mobilerun_agents import Run, tool

    entered = asyncio.Event()
    release = asyncio.Event()
    marker = "FAKE_INFLIGHT_942691"
    received = []

    @tool(sensitive_parameters=frozenset({"token"}))
    async def authorize(run: Run, token: str) -> None:
        """Pause a fake tool with a sensitive argument."""
        received.append(token)
        entered.set()
        await release.wait()

    selector = Selector(ToolCall("authorize", {"token": marker}), complete())
    handler = AgentWorkflow(selector).run(**await resources(), tools=[authorize])
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        inflight = handler.ctx.to_dict()
    finally:
        release.set()
    assert (await handler).success and received == [marker]
    assert inflight["workers"] and marker not in json.dumps(inflight)
    assert marker not in json.dumps(handler.ctx.to_dict())


@pytest.mark.anyio
async def test_pending_repair_error_is_masked_but_model_gets_raw_feedback(monkeypatch):
    from mobilerun_agents import Run, tool

    entered = asyncio.Event()
    release = asyncio.Event()
    marker = "FAKE_REPAIR_942691"

    @tool(sensitive_parameters=frozenset({"token"}))
    async def authorize(run: Run, token: str) -> None:
        """Track a fake secret for the repair attempt."""

    class Repairing(Selector):
        async def select(self, run):
            choice = await super().select(run)
            if len(self.requests) == 2:
                choice.parse_error = f"Invalid value {marker}"
            return choice

    selector = Repairing(ToolCall("authorize", {"token": marker}), complete(), complete())
    flow = AgentWorkflow(selector)
    remember = Run.remember

    async def pause_repair(self, message):
        if message.role == "user" and "Invalid value" in (message.content or ""):
            entered.set()
            await release.wait()
        await remember(self, message)

    monkeypatch.setattr(Run, "remember", pause_repair)
    handler = flow.run(**await resources(), tools=[authorize])
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        inflight = handler.ctx.to_dict()
    finally:
        release.set()
    assert (await handler).success
    assert f"Invalid value {marker}" in selector.requests[-1].memory
    assert inflight["workers"] and marker not in json.dumps(inflight)


@pytest.mark.anyio
async def test_screenshot_progress_is_not_excluded_before_capture():
    from mobilerun_agents import Run, tool

    args = await resources()
    phone = args["device"]
    phone.state._resolved_provider.get_state.return_value = UIState([], "", "", {}, 100, 100)
    actions = 0
    images = []

    @tool
    async def advance(run: Run) -> None:
        """Advance the fake screen."""
        nonlocal actions
        actions += 1

    async def screenshot():
        return f"image {max(0, actions - 1)}".encode()

    class Visual(ActionSelector):
        async def select(self, run):
            images.append(await run.screenshot())
            return ActionSelection(actions=[complete() if actions == 5 else ToolCall("advance")])

    phone.screenshot = screenshot
    handler = AgentWorkflow(Visual()).run(**args, tools=[advance])
    events = [event async for event in handler.stream_events()]
    assert (await handler).success and actions == 5
    assert images[:3] == [b"image 0", b"image 0", b"image 1"]
    assert not any(isinstance(event, Invalid) for event in events)


@pytest.mark.anyio
@pytest.mark.parametrize("productive", [True, False])
async def test_cycle_guard_only_counts_revisits_without_new_screens(productive):
    from mobilerun_agents import Run, tool

    args = await resources()
    provider = args["device"].state._resolved_provider
    screens = ["Hub", "A", "Hub", "B", "Hub", "C", "Hub", "D"] if productive else ["Hub", "A"] * 4
    actions = 0

    def render():
        title = screens[actions]
        provider.get_state.return_value = UIState([{"text": title}], title, "", {}, 100, 100)

    @tool
    async def navigate(run: Run) -> None:
        """Read the next item or go back to the hub."""
        nonlocal actions
        actions += 1
        render()

    render()
    selector = Selector(*[ToolCall("navigate") for _ in screens[1:]], complete())
    result = await AgentWorkflow(selector).run(**args, tools=[navigate])
    if productive:
        assert result.success and actions == 7
        assert all(
            "You have been on this screen" not in request.memory for request in selector.requests
        )
    else:
        assert not result.success and result.reason == "cycle"


@pytest.mark.anyio
async def test_raw_step_evidence_keeps_invalid_attempt_thought_after_repair():
    evidence = []
    blocker = "blocked: sign-in dialog hides prices"

    class Retry(ActionSelector):
        attempts = 0

        async def select(self, run):
            self.attempts += 1
            if self.attempts == 1:
                return ActionSelection(thought=blocker)
            return ActionSelection(actions=[complete()], thought="Report the partial result")

    handler = AgentWorkflow(Retry()).run(**await resources(), report_evidence=evidence.append)
    assert (await handler).success
    [steps] = evidence
    [step] = steps
    assert step.thought == f"{blocker}\nReport the partial result"


@pytest.mark.anyio
async def test_invalid_delegated_selection_repairs_without_reasking_primary():
    from mobilerun_agents.agent.action_selector import DelegatingSelector
    from mobilerun_agents.config.user_config import ToolsConfig

    class Primary(Selector):
        async def select(self, run):
            selection = await super().select(run)
            selection.confidence = 0.54
            return selection

    primary = Primary(*[ToolCall("wait")] * 3)

    class Delegate(ActionSelector):
        requests = []

        async def select(self, run):
            self.requests.append(await captured_turn(run))
            if len(self.requests) == 1:
                return ActionSelection(thought="No action yet.")
            if len(self.requests) == 2:
                return ActionSelection(actions=[ToolCall("wait")])
            return ActionSelection(actions=[complete()])

    delegate = Delegate()
    handler = AgentWorkflow(DelegatingSelector(primary, delegate, min_confidence=0.7)).run(
        **await resources(), tool_config=ToolsConfig(wait_duration=0.001)
    )
    assert (await handler).success
    assert len(primary.requests) == 2
    assert "call exactly one tool" in delegate.requests[1].memory
    assert len(delegate.requests) == 3


@pytest.mark.anyio
async def test_delegate_keeps_control_below_high_minimum_confidence():
    from mobilerun_agents.agent.action_selector import DelegatingSelector
    from mobilerun_agents.config.user_config import ToolsConfig

    class Primary(Selector):
        async def select(self, run):
            selection = await super().select(run)
            selection.confidence = 0.5 if len(self.requests) == 1 else 0.96
            return selection

    primary = Primary(complete(), complete())
    delegate = Selector(ToolCall("wait"), complete())
    handler = AgentWorkflow(DelegatingSelector(primary, delegate, min_confidence=0.98)).run(
        **await resources(), tool_config=ToolsConfig(wait_duration=0.001)
    )
    assert (await handler).success
    assert len(delegate.requests) == 2


@pytest.mark.anyio
@pytest.mark.parametrize("recovers", [True, False])
async def test_recursive_delegation_uses_existing_repair_budget(recovers):
    from mobilerun_agents.agent.action_selector import DelegatingSelector
    from mobilerun_agents.config.user_config import AgentConfig, DecisionConfig

    class Primary(Selector):
        async def select(self, run):
            selection = await super().select(run)
            selection.confidence = 0.1
            return selection

    primary = Primary(complete())
    delegate = Selector(ToolCall("delegate"), complete() if recovers else ToolCall("delegate"))
    config = AgentConfig(decision=DecisionConfig(max_response_repair_attempts=2))
    handler = AgentWorkflow(DelegatingSelector(primary, delegate)).run(
        **await resources(), config=config
    )
    events = [event async for event in handler.stream_events()]
    result = await handler
    assert result.success == recovers
    assert result.reason == ("complete" if recovers else "repair_attempts")
    assert len(primary.requests) == 1 and len(delegate.requests) == 2
    assert "you are the delegate; call one action tool" in delegate.requests[1].memory.lower()
    assert sum(type(event) is Invalid for event in events) == (1 if recovers else 2)


@pytest.mark.anyio
async def test_jev_takeover_requires_higher_confidence_and_sees_latest_delegate_intent():
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision
    from mobilerun_agents.agent.logs.diagnostics import Detail, private_values
    from mobilerun_agents.agent.logs.redaction import run_sensitive_values
    from mobilerun_agents.config.user_config import ToolsConfig

    intents = []
    first = "Open listing with fake-protected-value"
    latest = "Read the price"

    class Jev:
        max_options = None
        locates_points = False
        turns = iter(
            ((0.1, "wait"), (0.46, "wait"), (0.6, "tap_element"), (0.46, "wait"), (0.1, "wait"))
        )

        async def decide(self, request):
            intents.append(getattr(request, "intent", ""))
            confidence, name = next(self.turns)
            index = next(
                i for i, option in enumerate(request.options) if option.action.name == name
            )
            return Decision(option_index=index, confidence=confidence)

    class Delegate(Selector):
        debug_label = "llm:openai/fake"
        thoughts = iter((first, latest, "Finish"))

        async def select(self, run):
            selection = await super().select(run)
            selection.thought = next(self.thoughts)
            return selection

    delegate = Delegate(ToolCall("tap", {"x": 2, "y": 3}), ToolCall("wait"), complete())
    token = private_values.set(set())
    sensitive = run_sensitive_values.set({"fake-protected-value"})
    try:
        handler = AgentWorkflow(
            DelegatingSelector(DecisionSelector(Jev(), vision="off"), delegate, min_confidence=0.3)
        ).run(**await resources(), tool_config=ToolsConfig(wait_duration=0.001))
        events = [event async for event in handler.stream_events()]
        assert (await handler).success
    finally:
        run_sensitive_values.reset(sensitive)
        private_values.reset(token)

    assert [
        e.display_action.name for e in events if type(e) is StepFinished and e.display_action
    ] == [
        "tap",
        "wait",
        "tap_element",
        "wait",
        "complete",
    ]
    assert intents == ["", first, latest, latest, latest]
    details = [e.text for e in events if type(e) is Detail]
    assert "DecisionSelector → llm:openai/fake (takeover bar 0.5 not met: 0.46)" in details
    assert all("fake-protected-value" not in text for text in details)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "actions,parallel,diagnostic",
    [
        ([], True, "No tool call in reply"),
        ([complete(), complete()], False, "Expected one action, got 2"),
    ],
)
async def test_action_count_repair_has_specific_reason(actions, parallel, diagnostic):
    from mobilerun_agents.config.user_config import AgentConfig

    class Repairing(Selector):
        async def select(self, run):
            self.requests.append(await captured_turn(run))
            return ActionSelection(actions=actions if len(self.requests) == 1 else [complete()])

    selector = Repairing()
    handler = AgentWorkflow(selector).run(
        **await resources(), config=AgentConfig(parallel_tools=parallel)
    )
    events = [event async for event in handler.stream_events()]
    assert (await handler).success
    [invalid] = [event for event in events if type(event) is Invalid]
    assert invalid.diagnostic == diagnostic
    assert diagnostic in selector.requests[1].memory
    assert "call exactly one tool" in selector.requests[1].memory


@pytest.mark.anyio
async def test_one_console_dedupes_atomic_tool_logs_across_planning_children(caplog):
    import logging

    from mobilerun_agents.agent.logs.console import ConsoleLog
    from mobilerun_agents.agent.logs.diagnostics import private_values
    from mobilerun_agents.agent.planner import Finish, Plan
    from mobilerun_agents.agent.workflows.planning import PlanningWorkflow
    from tests.unit.agent.workflows.test_planning import ScriptedPlanner

    class AtomicSelector(ActionSelector):
        async def select(self, run):
            await run.registry.atomic_tools()
            return ActionSelection(actions=[complete()])

    args = await resources()
    provider = args["device"].state._resolved_provider

    def screen(count):
        return UIState(
            [{"index": n, "bounds": "0,0,20,40"} for n in range(1, count + 1)],
            "screen",
            "",
            {},
            100,
            100,
        )

    provider.get_state.side_effect = [screen(n) for n in (1, 1, 1, 1, 1, 2, 1, 1, 1)]
    console = ConsoleLog()
    caplog.set_level(logging.DEBUG, logger="mobilerun_agents.agent.logs.console")
    token = private_values.set(set())
    try:
        handler = PlanningWorkflow(
            ScriptedPlanner(
                *(Plan((goal,)) for goal in ("first", "second", "third", "fourth")),
                Finish(True, "done"),
            ),
            AtomicSelector(),
        ).run(**args)
        async for event in handler.stream_events():
            console.on_event(event)
        assert (await handler).success
    finally:
        private_values.reset(token)
    lines = [message.strip() for message in caplog.messages if "tools:" in message]
    assert sum("complete →" in message for message in lines) == 1
    assert [
        message.split(" → ", 1)[1].split(" (", 1)[0]
        for message in lines
        if "tap_element →" in message
    ] == [
        "1 element",
        "2 elements",
        "1 element",
    ]


@pytest.mark.anyio
async def test_cycle_reports_raw_step_evidence_only_to_the_child_callback():
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.logs.diagnostics import private_values
    from mobilerun_agents.agent.workflows.events import GuardDetail

    args = await resources()
    provider = args["device"].state._resolved_provider
    secret = "FAKE token / run evidence"
    visits = 0
    evidence = []

    def render():
        title = ("Search", "Results", "Product")[visits % 3]
        provider.get_state.return_value = UIState(
            [{"index": 1, "text": title}],
            title,
            "",
            {"packageName": "shop", "activityName": "Main"},
            100,
            100,
        )

    @tool(sensitive_parameters=frozenset({"token"}))
    async def navigate(run: Run, token: str) -> None:
        """Open the next fake screen."""
        nonlocal visits
        visits += 1
        render()

    class Loop(Selector):
        async def select(self, run):
            choice = await super().select(run)
            choice.thought = f"Try again with {secret}"
            return choice

    render()
    token = private_values.set(set())
    try:
        selector = Loop(*[ToolCall("navigate", {"token": secret}) for _ in range(12)])
        handler = AgentWorkflow(selector).run(
            **args, tools=[navigate], report_evidence=evidence.append
        )
        events = [event async for event in handler.stream_events()]
        result = await handler
    finally:
        private_values.reset(token)
    assert (result.success, result.reason, result.steps) == (False, "cycle", 11)
    [steps] = evidence
    assert [step.step for step in steps] == list(range(1, 13))
    assert all(step.thought == f"Try again with {secret}" for step in steps[:-1])
    assert (steps[0].package, steps[0].activity, steps[0].title) == ("shop", "Main", "Search")
    assert steps[-1].thought is None and steps[-1].fingerprint == steps[2].fingerprint
    assert len({step.fingerprint for step in steps}) == 3
    assert any(
        "You have been on this screen 3 times" in request.memory for request in selector.requests
    )
    assert any(
        isinstance(event, GuardDetail) and event.text.startswith("cycle:") for event in events
    )
    assert secret not in repr(events) and secret not in json.dumps(handler.ctx.to_dict())


@pytest.mark.anyio
async def test_encoded_secret_is_absent_from_public_thought_and_console(caplog):
    import logging
    from urllib.parse import quote_plus

    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.logs.console import ConsoleLog
    from mobilerun_agents.agent.logs.diagnostics import Detail, private_values

    secret = "FAKE password / test=84927"
    encoded = quote_plus(secret)

    @tool(sensitive_parameters=frozenset({"token"}))
    async def confidential(run: Run, token: str) -> None:
        """Use a fake private token."""

    class Echo(Selector):
        async def select(self, run):
            choice = await super().select(run)
            if len(self.requests) == 2:
                choice.thought = f"Echo {encoded}"
            return choice

    token = private_values.set(set())
    try:
        caplog.set_level(logging.DEBUG)
        handler = AgentWorkflow(Echo(ToolCall("confidential", {"token": secret}), complete())).run(
            **await resources(), tools=[confidential]
        )
        console = ConsoleLog()
        thoughts = []
        async for event in handler.stream_events():
            console.on_event(event)
            if isinstance(event, Detail) and event.section == "thought":
                thoughts.append(event.text)
        assert (await handler).success
        assert thoughts == ["Echo ***"]
        assert "Echo ***" in caplog.text and encoded not in caplog.text
    finally:
        private_values.reset(token)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "changed", ["contentDescription", "resourceId", "isChecked", "isSelected", None]
)
async def test_swipe_guard_distinguishes_accessibility_progress(changed):
    args = await resources()
    device = args["device"]
    device._supported_operations = {"swipe"}
    provider = device.state._resolved_provider
    provider.requires_active_contract_for_coords = False
    count = 0

    def render():
        node = {"index": 1, "className": "ImageView", "bounds": "0,0,80,80"}
        if changed:
            node[changed] = bool(count % 2) if changed.startswith("is") else f"Item {count}"
        provider.get_state.return_value = UIState([node], "screen", "", {}, 100, 100)

    async def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000) -> None:
        nonlocal count
        count += 1
        render()

    render()
    device.swipe = swipe
    call = ToolCall("swipe", {"x1": 5, "y1": 80, "x2": 5, "y2": 20, "duration_ms": 500})
    # Two boolean values can genuinely cycle; stop before either is visited four times.
    handler = AgentWorkflow(Selector(*[call] * 5, complete())).run(**args)
    events = [event async for event in handler.stream_events()]
    result = await handler
    if changed:
        assert result.success and count == 5
        assert not any(isinstance(e, Invalid) for e in events)
    else:
        assert not result.success and result.reason == "stuck" and count == 4

import asyncio
import base64
from unittest.mock import AsyncMock, NonCallableMock

import pytest
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel
from workflows import Context
from workflows.context.serializers import JsonSerializer
from workflows.errors import WorkflowCancelledByUser

from mobilerun_agents.agent.action_selector.base import ActionSelector, Selection
from mobilerun_agents.agent.events import Acted, Finished, Observed, Subgoal
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.memory import MemoryDetail
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory.memory import _RunMemory
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.agent.tools.base import tool
from mobilerun_agents.agent.tools.helpers.images import ImageStore
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.agent.workflows.planning import PlanningWorkflow, _clip
from mobilerun_agents.agent.workflows.resources import Services, run_resources
from mobilerun_agents.config.user_config import AgentConfig, ToolsConfig
from tests.unit.device_fixtures import MemoryDevice


class FakePlanner(Planner):
    def __init__(self, *turns):
        self.turns = iter(turns)
        self.states = []
        self.prompt_inputs = []

    def bind(self, output_schema, repair, observe_screen):
        self.prompt_inputs.append((output_schema, repair, observe_screen))
        return self

    async def plan(self, ctx):
        self.states.append(await ctx.store.get_state())
        return next(self.turns)


class FakeSelector(ActionSelector):
    def __init__(self, *calls):
        self.calls = iter(calls)
        self.memories = []
        self.goals = []

    async def select(self, ctx, device, memory, registry):
        self.goals.append((await ctx.store.get_state()).goal)
        self.memories.append(memory)
        return Selection(calls=[next(self.calls)], thought="Blocked: Sign-in dialog hides prices.")


class Output(BaseModel):
    price: int


def complete(message="done", success=True):
    return ToolSelection(
        tool_id="complete",
        tool_name="complete",
        tool_kwargs={"success": success, "message": message},
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def phone():
    device = MemoryDevice()
    provider = NonCallableMock(spec=StateProvider)
    provider.get_state = AsyncMock(
        return_value=UIState([], "Prices shown", "", {"packageName": "shop"}, 100, 100)
    )
    device._set_state_provider(provider)
    device.screenshot = AsyncMock(
        return_value=base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aUQ0AAAAASUVORK5CYII="
        )
    )
    return device


@pytest.fixture
def make_workflow(phone, tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path))

    @tool
    async def tick() -> str:
        """Record a fake tick without contacting the device."""
        return "tick"

    @tool(sensitive_parameters=frozenset({"text"}))
    async def protect(text: str) -> str:
        """Identify a protected test value."""
        return "protected"

    async def make(planner, executor, **options):
        registry = build_registry(phone, [tick, protect], ToolsConfig())
        services = Services(registry, ImageStore("run", keep=True))
        memory = Memory.from_defaults(tokenizer_fn=str.split)
        manager = await run_resources(phone, memory, services)
        return PlanningWorkflow(planner, executor, resource_manager=manager, **options)

    return make


@pytest.mark.anyio
async def test_planner_receives_child_repair_exhaustion_from_jev(make_workflow):
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.config.user_config import DecisionConfig
    from tests.unit.agent.action_selector.test_delegating_selector import Model, Stub

    planner = FakePlanner(Plan(("Find price",)), Finish(False, "No reliable action"))
    decision = Model(("wait", 0.2), ("wait", 0.2))
    delegate = Stub()
    selector = DelegatingSelector(DecisionSelector(decision, "off", min_confidence=0.7), delegate)
    workflow = await make_workflow(
        planner,
        selector,
        config=AgentConfig(decision=DecisionConfig(max_response_repair_attempts=1)),
    )
    result = await workflow.run(goal="Find price")
    assert result.message == "No reliable action"
    child = planner.states[1].results[0]
    assert not child.success and child.reason == "repair_attempts" and not delegate.calls


@pytest.mark.anyio
async def test_delegation_retry_and_intent_are_local_to_each_planning_child(make_workflow):
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
    from tests.unit.agent.action_selector.test_delegating_selector import Model, Stub, pick

    class Primary(DecisionSelector):
        async def select(self, ctx, device, memory, registry, **kwargs):
            state = await ctx.store.get_state()
            assert not state.retry_with_delegate
            return await super().select(ctx, device, memory, registry, **kwargs)

    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Finish(True, "Done"))
    model = Model(("delegate", 0.9), ("complete", 0.9), ("complete", 0.9))
    delegate = Stub(
        InvalidResponse("Bad reply"), Selection(calls=[pick("wait")], thought="First child only")
    )
    workflow = await make_workflow(planner, DelegatingSelector(Primary(model, "off"), delegate))
    result = await workflow.run(goal="Two children")

    assert result.success
    assert [request.intent for request in model.requests] == ["", "First child only", ""]
    assert len(delegate.calls) == 2


@pytest.mark.anyio
async def test_planner_reconsiders_each_subgoal_and_saves_one_ordered_action_history(make_workflow):
    planner = FakePlanner(
        Plan(("Look up price", "Unused remainder")),
        Plan(("Dismiss sign-in",)),
        Finish(True, "The price is 42", {"price": 42}),
    )
    selector = FakeSelector(complete("Could not load prices", False), complete("Now visible"))
    flow = await make_workflow(planner, selector, output=Output)
    handler = flow.run(goal="Find the price")
    events = [event async for event in handler.stream_events()]
    result = await handler

    assert result.output == {"price": 42}
    assert [
        (event.text, event.result and event.result.message)
        for event in events
        if isinstance(event, Subgoal)
    ] == [
        ("Look up price", None),
        ("Look up price", "Could not load prices"),
        ("Dismiss sign-in", None),
        ("Dismiss sign-in", "Now visible"),
    ]
    assert [event.step for event in events if isinstance(event, Observed)] == [1, 2]
    assert selector.goals == ["Look up price", "Dismiss sign-in"]
    assert all(isinstance(memory, Memory) for memory in selector.memories)
    assert selector.memories[0] is not selector.memories[1]
    assert planner.states[1].plan == ["Unused remainder"]
    failed = planner.states[1].results[0]
    assert not failed.success and failed.reason == "complete"
    assert "Sign-in dialog hides prices." in failed.summary and "shop" in failed.summary
    assert [event.action.step for event in events if isinstance(event, Acted)] == [1, 2]
    assert [event.steps for event in events if isinstance(event, Finished)] == [2]
    serializer = JsonSerializer()
    saved = Context.from_dict(
        flow, handler.ctx.to_dict(serializer=serializer), serializer=serializer
    )
    actions = (await saved.store.get_state()).actions
    assert [(record.step, record.call.tool_kwargs["message"]) for record in actions] == [
        (1, "Could not load prices"),
        (2, "Now visible"),
    ]
    paths = [record.before.screenshot.path for record in actions]
    assert paths[0] != paths[1] and paths[0].parent == paths[1].parent


@pytest.mark.anyio
async def test_console_keeps_memory_for_second_child_when_first_makes_no_action(
    make_workflow, caplog
):
    class EmptyFirstSelector(FakeSelector):
        async def select(self, ctx, device, memory, registry):
            goal = (await ctx.store.get_state()).goal
            ctx.write_event_to_stream(
                MemoryDetail(
                    section="memory",
                    text=goal,
                    blocks={"notes": goal},
                    counts="counts",
                    last_steps=[],
                )
            )
            if goal == "first child":
                return Selection()
            return Selection(calls=[complete()])

    planner = FakePlanner(Plan(("first child",)), Plan(("second child",)), Finish(True, "done"))
    flow = await make_workflow(planner, EmptyFirstSelector())
    handler = flow.run(goal="root")
    console = ConsoleLog(Redactor())
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    async for event in handler.stream_events():
        console.event(event)
    await handler
    assert "notes: second child" in caplog.text


def test_clip_scans_long_screen_once_and_backs_up_across_overlapping_spans():
    class CountingRedactor(Redactor):
        calls = 0

        def text(self, value: str) -> str:
            self.calls += 1
            return super().text(value)

        def masked_spans(self, value: str) -> list[tuple[int, int]]:
            self.calls += 1
            return super().masked_spans(value)

    redact = CountingRedactor()
    registered = f"{'x' * 98} suffix"
    redact.add(registered)
    screen = f"prefix Bearer {registered} " + "line " * 200_000
    assert _clip(screen, 115, redact) == "prefix "
    assert redact.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "expected"), [("", True), ("Goal achieved.", False), ("Battery at 100%", False)]
)
async def test_only_missing_description_requests_observation(
    make_workflow, phone, message, expected
):
    planner = FakePlanner(Plan(("Read battery",)), Finish(True, "Done"))
    flow = await make_workflow(planner, FakeSelector(complete(message)))
    assert (await flow.run(goal="Check battery")).success
    state = planner.states[1]
    observe_screen = planner.prompt_inputs[1][2]
    assert observe_screen == expected
    if observe_screen:
        assert state.current.screenshot.path.read_bytes() == await phone.screenshot()


@pytest.mark.anyio
async def test_later_child_protection_clears_earlier_image_references(make_workflow):
    planner = FakePlanner(
        Plan(("Look",)), Plan(("Identify protected value",)), Finish(True, "Done")
    )
    protected = ToolSelection(
        tool_id="protect", tool_name="protect", tool_kwargs={"text": "fake-secret"}
    )
    flow = await make_workflow(planner, FakeSelector(complete(), protected, complete("")))
    handler = flow.run(goal="Review")
    assert (await handler).success
    path = planner.states[1].actions[0].before.screenshot.path
    assert not path.exists()
    assert planner.prompt_inputs[2][2] and planner.states[2].current.screenshot is None
    assert all(record.before.screenshot is None for record in planner.states[2].actions)


@pytest.mark.anyio
async def test_child_budget_allows_replan_but_total_budget_prevents_another_child(make_workflow):
    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Plan(("Never starts",)))
    selector = FakeSelector(
        ToolSelection(tool_id="tick", tool_name="tick", tool_kwargs={}), complete()
    )
    config = AgentConfig(planning={"subgoal_steps": 1})
    flow = await make_workflow(planner, selector, config=config)
    handler = flow.run(goal="Recover", max_steps=2)
    events = [event async for event in handler.stream_events()]
    assert (await handler).reason == "max_steps"
    assert [event.step for event in events if isinstance(event, Observed)] == [1, 2]
    assert [event.steps for event in events if isinstance(event, Finished)] == [2]
    assert planner.states[1].results[0].reason == "max_steps"
    assert selector.goals == ["First", "Second"]


@pytest.mark.anyio
async def test_planner_turn_budget_is_independent_of_child_steps(make_workflow):
    planner = FakePlanner(Plan(("Only child",)))
    flow = await make_workflow(
        planner, FakeSelector(complete()), config=AgentConfig(planning={"max_planner_turns": 1})
    )
    handler = flow.run(goal="Check")
    assert (await handler).reason == "max_planner_turns"
    assert len(planner.states) == (await handler.ctx.store.get_state()).executor_steps == 1


@pytest.mark.anyio
async def test_planner_can_finish_at_exact_executor_limit(make_workflow):
    planner = FakePlanner(Plan(("Read value",)), Finish(True, "Found it"))
    flow = await make_workflow(planner, FakeSelector(complete("Read 12")))
    assert (await flow.run(goal="Find value", max_steps=1)).success
    assert planner.states[1].results[0].message == "Read 12"


@pytest.mark.anyio
async def test_typed_planner_output_repairs_without_rerunning_executor(make_workflow):
    planner = FakePlanner(
        Finish(True, "Read", {"price": "42"}), Finish(True, "Read", {"price": 42})
    )
    selector = FakeSelector()
    flow = await make_workflow(planner, selector, output=Output)
    assert (await flow.run(goal="Find price")).output == {"price": 42}
    assert selector.goals == []
    schema, repair, _ = planner.prompt_inputs[1]
    assert schema["properties"]["price"]["type"] == "integer"
    assert "output is invalid" in repair


@pytest.mark.anyio
async def test_parent_cancellation_stops_running_child(make_workflow):
    started = asyncio.Event()
    ended = asyncio.Event()

    class Waiting(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            started.set()
            try:
                await asyncio.Future()
            finally:
                ended.set()

    flow = await make_workflow(FakePlanner(Plan(("Wait",))), Waiting())
    handler = flow.run(goal="Cancel")
    await asyncio.wait_for(started.wait(), timeout=5)
    await handler.cancel_run()
    with pytest.raises(WorkflowCancelledByUser):
        await handler
    await asyncio.wait_for(ended.wait(), timeout=5)


@pytest.mark.anyio
@pytest.mark.parametrize("batch", ["parallel", "sequential", "complete_only"])
async def test_planner_gets_post_action_tree_without_redundant_refresh(make_workflow, phone, batch):
    tick = ToolSelection(tool_id="tick", tool_name="tick", tool_kwargs={})
    completion = complete("")
    turns = iter(
        {
            "parallel": [[tick, completion]],
            "sequential": [[tick], [completion]],
            "complete_only": [[completion]],
        }[batch]
    )
    before = UIState([{"text": "Before"}], "Before", "", {}, 100, 100)
    after = UIState([{"text": "After"}], "After", "", {}, 100, 100)
    provider = phone._state_provider_override
    provider.get_state.side_effect = [before, before, after]

    async def screenshot():
        return phone.state.current.formatted_text.encode()

    phone.screenshot.side_effect = screenshot

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            return Selection(calls=next(turns))

    planner = FakePlanner(Plan(("Advance",)), Finish(True, "done"))
    workflow = await make_workflow(planner, Selector())
    assert (await workflow.run(goal="Advance")).success

    state = planner.states[1]
    expected = "Before" if batch == "complete_only" else "After"
    assert state.current.ui.formatted_text == expected
    assert state.actions[0].before.ui.formatted_text == "Before"
    assert provider.get_state.await_count == (2 if batch == "complete_only" else 3)
    assert state.current.screenshot.path.read_bytes() == expected.encode()
    assert phone.screenshot.await_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "position", ["crossing", "ends_at_cut", "starts_at_cut", "unprotected", "encoded_crossing"]
)
async def test_failure_summary_never_clips_through_protected_values(make_workflow, phone, position):
    secret = "fixture alpha" if position == "encoded_crossing" else "fixture-secret"
    marker = "fixture%20alpha" if position == "encoded_crossing" else secret
    remaining = {
        "crossing": 4,
        "ends_at_cut": len(marker),
        "starts_at_cut": 0,
        "unprotected": 4,
        "encoded_crossing": 12,
    }[position]
    thought_prefix = "x" * (240 - remaining)
    screen_prefix = "y" * (120 - remaining)
    provider = phone._state_provider_override
    provider.get_state.return_value = UIState(
        [{"text": "screen"}], screen_prefix + marker + " tail", "", {}, 100, 100
    )

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            if position != "unprotected":
                registry.redactor.add(secret)
            return Selection(
                calls=[complete("blocked", False)], thought=thought_prefix + marker + " tail"
            )

    planner = FakePlanner(Plan(("Advance",)), Finish(False, "done"))
    workflow = await make_workflow(planner, Selector())
    await workflow.run(goal="Advance")

    summary = planner.states[1].results[0].summary
    suffix = "" if position in {"crossing", "encoded_crossing"} else marker[:remaining]
    assert f"- {thought_prefix}{suffix}\n" in summary
    assert f" — {screen_prefix}{suffix} ×1\n" in summary


@pytest.mark.anyio
@pytest.mark.parametrize("secret", ["fixture-a. fixture-b", "fixture-a\nfixture-b"])
async def test_failure_summary_never_splits_protected_thoughts(make_workflow, secret):
    thought = f"Blocked by {secret} while signing in."

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            registry.redactor.add(secret)
            return Selection(calls=[complete("blocked", False)], thought=thought)

    planner = FakePlanner(Plan(("Advance",)), Finish(False, "done"))
    workflow = await make_workflow(planner, Selector())
    await workflow.run(goal="Advance")

    summary = planner.states[1].results[0].summary
    redact = Redactor()
    redact.add(secret)
    assert "fixture-a" not in redact.text(summary)
    assert summary.count(f"- {thought}\n") == 2


@pytest.mark.anyio
@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancellation"])
async def test_planning_closes_each_child_memory(make_workflow, monkeypatch, caplog, exit_kind):
    started = asyncio.Event()
    closed = asyncio.Event()
    used = []
    stopped = []
    disposed = []
    original_close = _RunMemory.aclose

    async def close(memory):
        await original_close(memory)
        disposed.append((memory, any(item is memory for item in stopped)))
        closed.set()
        raise RuntimeError("fixture close failure")

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            used.append(memory)
            started.set()
            try:
                if exit_kind == "failure":
                    raise ValueError("fixture child failure")
                if exit_kind == "cancellation":
                    await asyncio.Future()
                return Selection(calls=[complete()])
            finally:
                stopped.append(memory)

    monkeypatch.setattr(_RunMemory, "aclose", close)
    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Finish(True, "done"))
    workflow = await make_workflow(planner, Selector())
    run = workflow.run(goal="goal")
    await asyncio.wait_for(started.wait(), 5)
    if exit_kind == "cancellation":
        await run.cancel_run()
        with pytest.raises(WorkflowCancelledByUser):
            await run
    elif exit_kind == "failure":
        with pytest.raises(ValueError, match="fixture child failure"):
            await run
    else:
        assert (await run).success

    await asyncio.wait_for(closed.wait(), 1)
    assert len(used) == len(disposed) == (2 if exit_kind == "success" else 1)
    assert len({id(memory) for memory in used}) == len(used)
    assert all(
        closed_memory is memory and stopped_first
        for (closed_memory, stopped_first), memory in zip(disposed, used, strict=True)
    )
    assert "fixture close failure" not in caplog.text

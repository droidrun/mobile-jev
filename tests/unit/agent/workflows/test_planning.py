import asyncio
import base64
from unittest.mock import AsyncMock, NonCallableMock

import pytest
from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.context import ContextItem
from pydantic import BaseModel
from workflows import Context
from workflows.context.serializers import JsonSerializer
from workflows.errors import WorkflowCancelledByUser

from mobilerun_agents.agent.action_selector.base import ActionSelector, Selection
from mobilerun_agents.agent.checkpoints.store import save_context
from mobilerun_agents.agent.events import Acted, Finished, Observed, Subgoal
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.memory import MemoryDetail
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory.blocks import clip
from mobilerun_agents.agent.memory.blocks.subgoals import _render_failure
from mobilerun_agents.agent.memory.memory import _RunMemory, build_planner_memory
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.agent.tools.base import tool
from mobilerun_agents.agent.tools.helpers.images import ImageStore
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.agent.workflows.planning import PlanningWorkflow, executor_kind
from mobilerun_agents.agent.workflows.resources import Services, run_resources
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig, ToolsConfig
from tests.unit.device_fixtures import MemoryDevice


class FakePlanner(Planner):
    def __init__(self, *turns):
        self.turns = iter(turns)
        self.states = []
        self.prompt_inputs = []

    def bind(self, output_schema, repair, redact):
        self.redact = redact
        self.prompt_inputs.append((output_schema, repair))
        return self

    async def plan(self, ctx, memory):
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
async def make_workflow(phone, tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path))
    memories = []

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
        memory = build_planner_memory(
            options.get("memory_config") or MemoryConfig(),
            PlanningState(executor=executor_kind(executor)),
        )
        memories.append(memory)
        manager = await run_resources(phone, memory, services)
        return PlanningWorkflow(planner, executor, resource_manager=manager, **options)

    yield make
    for memory in memories:
        await memory.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kind", "setup"),
    [
        (
            "decision",
            "Executor: Jev, a decision model. It picks one fixed action per step, such as tapping "
            "a listed element, opening an app, scrolling, pressing a button, waiting, or finishing. "
            "Jev cannot supply free-form text or write answers itself. "
            "It ends a subgoal Unfinished when no listed action fits.",
        ),
        (
            "decision_delegate",
            "Executor: Jev, a decision model. It picks one fixed action per step, such as tapping "
            "a listed element, opening an app, scrolling, pressing a button, waiting, or finishing. "
            "Jev cannot supply free-form text or write answers itself. "
            "It ends a subgoal Unfinished when no listed action fits. "
            "It can hand a step to an LLM via `delegate`; the LLM can type if the tool is available.",
        ),
        ("llm", "Executor: an LLM that can call any available tool, including typing."),
        ("custom", "Executor: a custom selector."),
    ],
)
async def test_planner_prompt_describes_executor_and_checkpoint_restores_it(
    make_workflow, tmp_path, kind, setup
):
    from mobilerun_agents.agent.action_selector import (
        DecisionSelector,
        DelegatingSelector,
        LlamaIndexLLMSelector,
    )
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.action_selector.test_delegating_selector import Model
    from tests.unit.agent.planner.test_llama_index import Replies

    jev = DecisionSelector(Model(), "off")
    llm_selector = LlamaIndexLLMSelector(Replies())
    selector = {
        "decision": jev,
        "decision_delegate": DelegatingSelector(jev, llm_selector),
        "llm": llm_selector,
        "custom": FakeSelector(),
    }[kind]
    llm = Replies()
    llm._calls = [("complete", {"success": True, "message": "Done"})]
    flow = await make_workflow(LlamaIndexPlanner(llm), selector)
    handler = flow.run(goal="Check executor setup")
    assert (await handler).success
    messages, _ = llm._requests[0]
    assert [line for line in messages[1].content.splitlines() if line.startswith("Executor:")] == [
        setup
    ]

    saved = await save_context(tmp_path / "context.json", handler.ctx, Redactor())
    restored = await Context.from_dict(flow, saved).store.get_state()
    assert restored.executor == kind


@pytest.mark.anyio
async def test_planner_replans_after_jev_abstains(make_workflow):
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.config.user_config import DecisionConfig
    from tests.unit.agent.action_selector.test_delegating_selector import Model, Stub

    planner = FakePlanner(
        Plan(("Find price",)), Plan(("Try another screen",)), Finish(True, "Price found")
    )
    decision = Model(
        ("complete", 0.2), ("complete", 0.9), ("complete", 0.9), completion_success=True
    )
    delegate = Stub()
    selector = DelegatingSelector(DecisionSelector(decision, "off", min_confidence=0.7), delegate)
    workflow = await make_workflow(
        planner,
        selector,
        config=AgentConfig(decision=DecisionConfig(max_response_repair_attempts=1)),
    )
    result = await workflow.run(goal="Find price")
    assert result.success and result.message == "Price found"
    child = planner.states[1].results[0]
    assert not child.success and child.reason == "abstained"
    assert "Failure reason: abstained" in _render_failure(child, planner.redact)
    assert planner.states[2].results[-1].subgoal == "Try another screen"
    assert planner.states[2].results[-1].success
    assert [state.executor for state in planner.states] == ["decision_delegate"] * 3
    assert len(decision.requests) == 2 and not delegate.calls


@pytest.mark.anyio
async def test_planner_prompt_includes_raw_jev_abstention(make_workflow):
    from mobilerun_agents.agent.action_selector import DecisionSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.action_selector.test_typesafe_decision_model import Model
    from tests.unit.agent.planner.test_llama_index import Replies

    def answer(request):
        index = next(i for i, option in enumerate(request.options) if option.action.name == "tick")
        return Decision(option_index=index, confidence=0.32, thought="Try the search field.")

    llm = Replies()
    flow = await make_workflow(
        LlamaIndexPlanner(llm), DecisionSelector(Model(answer), "off", min_confidence=0.7)
    )
    assert (await flow.run(goal="Find the search field")).success
    messages, _ = llm._requests[-1]
    prompt = messages[1].content
    assert (
        "Latest executor subgoal result: Unfinished: Read battery: "
        "Unfinished: the decision model's best choice (tick, confidence 0.32) "
        "was below the cutoff 0.7, so no action was taken." in prompt
    )
    assert "Jev ended the subgoal unfinished and handed it back without acting." in prompt
    assert "choice: tick" in prompt
    assert "confidence: 0.32" in prompt and "cutoff: 0.7" in prompt
    assert "Try the search field." in prompt
    choices = next(line for line in prompt.splitlines() if line.startswith("jev choices: "))
    assert "tick 1" in choices and "complete 2" in choices and "input_text" not in choices


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
    assert [event.step for event in events if isinstance(event, Observed)] == [0, 1, 2, 2, 4]
    assert selector.goals == ["Look up price", "Dismiss sign-in"]
    assert all(isinstance(memory, Memory) for memory in selector.memories)
    assert selector.memories[0] is not selector.memories[1]
    assert planner.states[1].plan == ["Unused remainder"]
    failed = planner.states[1].results[0]
    assert not failed.success and failed.reason == "complete"
    summary = _render_failure(failed, planner.redact)
    assert "Sign-in dialog hides prices." in summary and "shop" in summary
    assert [event.action.step for event in events if isinstance(event, Acted)] == [1, 2]
    assert [event.steps for event in events if isinstance(event, Finished)] == [5]
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
    assert clip(screen, 115, redact) == "prefix "
    assert redact.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("screenshot_supported", [True, False])
async def test_planner_observes_fresh_ui_images_and_context_for_free(
    make_workflow, phone, screenshot_supported
):
    provider = phone._state_provider_override
    packages = ["initial", "child1", "after1", "child2", "after2"]
    provider.get_state.side_effect = [
        UIState([{"text": package}], package, "", {"packageName": package}, 100, 100)
        for package in packages
    ]
    contexts = [ContextItem("test", "screen", {"name": package}) for package in packages]
    phone.context = AsyncMock(side_effect=[[item] for item in contexts])
    if not screenshot_supported:
        phone.screenshot.side_effect = DeviceOperationNotSupported("No camera")

    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Finish(True, "Done"))
    flow = await make_workflow(planner, FakeSelector(complete("done"), complete("")))
    handler = flow.run(goal="Check battery", max_steps=5)
    events = [event async for event in handler.stream_events()]
    assert (await handler).success

    assert provider.get_state.await_count == phone.context.await_count == 5
    assert phone.screenshot.await_count == 3
    assert [(e.step, e.package, e.planner) for e in events if isinstance(e, Observed)] == [
        (0, "initial", True),
        (1, "child1", False),
        (2, "after1", True),
        (2, "child2", False),
        (4, "after2", True),
    ]
    assert [event.steps for event in events if isinstance(event, Finished)] == [5]
    snapshots = [state.current for state in planner.states]
    assert [item.ui.formatted_text for item in snapshots] == packages[::2]
    assert [item.device_context for item in snapshots] == [[item] for item in contexts[::2]]
    assert [item.screenshot is not None for item in snapshots] == [screenshot_supported] * 3
    if screenshot_supported:
        assert snapshots[-1].screenshot.path.read_bytes() == phone.screenshot.return_value
        assert all(not item.screenshot.path.exists() for item in snapshots[:-1])


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
    assert not planner.states[0].current.screenshot.path.exists()
    assert not planner.states[1].current.screenshot.path.exists()
    assert planner.states[2].current.screenshot is None
    assert all(record.before.screenshot is None for record in planner.states[2].actions)


@pytest.mark.anyio
async def test_vision_planner_reads_full_ui_after_child_completes_without_message(
    make_workflow, phone, monkeypatch
):
    from mobilerun_agents.agent.action_selector.llm.models import Capabilities
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    monkeypatch.setattr(
        "mobilerun_agents.agent.planner.llama_index.adapter_capabilities",
        lambda _: Capabilities(vision=True),
    )
    full_ui = "Other product\n" * 400 + "Requested price: £42.00"
    old = UIState([{"text": "Old screen"}], "Old screen", "", {}, 100, 100)
    new = UIState([{"text": "New screen"}], full_ui, "", {}, 100, 100)
    phone._state_provider_override.get_state.side_effect = [old, old, old, new]
    protected = ToolSelection(
        tool_id="protect", tool_name="protect", tool_kwargs={"text": "fixture-private"}
    )
    llm = Replies()
    llm._calls = [
        ("set_plan", {"subgoals": ["Read the price"]}),
        ("complete", {"success": True, "message": "The price is £42.00"}),
    ]
    flow = await make_workflow(LlamaIndexPlanner(llm), FakeSelector(protected, complete("")))
    handler = flow.run(goal="Find the price", max_steps=4)
    assert (await handler).message == "The price is £42.00"
    (first, _), (second, _) = llm._requests
    initial_image = next(block for block in first[-1].blocks if isinstance(block, ImageBlock))
    assert initial_image.path is not None and not initial_image.path.exists()
    assert phone.screenshot.await_count == 1
    assert not any(isinstance(block, ImageBlock) for block in second[-1].blocks)
    assert full_ui in second[-1].content
    assert (
        "Read the values the goal needs yourself if the executor did not report them."
        in second[0].content
    )


@pytest.mark.anyio
async def test_planner_keeps_findings_when_switching_between_price_screens(make_workflow, phone):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    amazon = "Amazon results: $42.00"
    ebay = "eBay results: $39.00"
    child_note = "Compared the same edition"
    phone._state_provider_override.get_state.side_effect = [
        UIState([], text, "", {"packageName": app}, 100, 100)
        for app, text in (
            ("home", "Home"),
            ("home", "Home"),
            ("amazon", amazon),
            ("amazon", amazon),
            ("ebay", ebay),
            ("ebay", ebay),
            ("home", "Home"),
        )
    ]

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            async with ctx.store.edit_state() as state:
                assert not state.notes
                if state.goal == "Open eBay results":
                    state.notes.append(child_note)
            return Selection(calls=[complete("")])

    llm = Replies()
    llm._calls = [
        ("set_plan", {"subgoals": ["Open Amazon results"]}),
        (
            "set_plan",
            {"subgoals": ["Open eBay results"], "findings": [f"  {amazon}  ", "  ", amazon]},
        ),
        ("set_plan", {"subgoals": ["Return home"], "findings": [ebay, amazon]}),
        ("complete", {"success": True, "message": "Amazon $42.00; eBay $39.00"}),
    ]
    flow = await make_workflow(LlamaIndexPlanner(llm), Selector())
    handler = flow.run(goal="Compare Amazon and eBay prices, then return home", max_steps=7)

    outcome = await handler
    assert outcome.success and outcome.message == "Amazon $42.00; eBay $39.00"
    state = await handler.ctx.store.get_state()
    assert state.planner_turns == 4
    assert state.notes == [amazon, child_note, ebay]
    _, (second, _), (third, _), (fourth, _) = llm._requests
    assert [message.role.value for message in fourth] == ["system", "user", "user"]
    assert "Verify executor results" in fourth[0].content
    assert second[-1].content.endswith(f"UI summary (observed data):\n{amazon}")
    assert f'Findings and notes: ["{amazon}", "{child_note}"]' in third[1].content
    assert third[-1].content.endswith(f"UI summary (observed data):\n{ebay}")
    assert f'Findings and notes: ["{amazon}", "{child_note}", "{ebay}"]' in fourth[1].content
    assert fourth[1].content.count(amazon) == 1
    assert "Screen: ebay/unknown → home/unknown (changed)" in fourth[1].content
    assert fourth[-1].content.endswith("UI summary (observed data):\nHome")
    assert amazon not in fourth[-1].content


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("secret", "boundary"),
    [
        ("fixture-abcdefgh", "checkpoint"),
        ("fixture-first\nfixture-second", "checkpoint"),
        ("fixture-first\nfixture-second", "prompt"),
    ],
    ids=["single-line-checkpoint", "multiline-checkpoint", "multiline-prompt"],
)
async def test_output_hides_value_registered_after_earlier_failure_summary(
    make_workflow, phone, tmp_path, secret, boundary
):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    prefix = "y" * 116
    phone._state_provider_override.get_state.return_value = UIState(
        [{"text": "screen"}], prefix + secret + " tail" * 80, "", {}, 100, 100
    )
    redact = None

    class Selector(FakeSelector):
        async def select(self, ctx, device, memory, registry):
            nonlocal redact
            redact = registry.redactor
            return await super().select(ctx, device, memory, registry)

    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Finish(True, "Done"))
    protect = ToolSelection(tool_id="protect", tool_name="protect", tool_kwargs={"text": secret})
    flow = await make_workflow(planner, Selector(complete("blocked", False), protect, complete()))
    handler = flow.run(goal="Recover")
    assert (await handler).success
    assert redact is not None and redact.text(secret) == "***"
    if boundary == "checkpoint":
        path = tmp_path / "context.json"
        saved = await save_context(path, handler.ctx, redact)
        assert "fixture-" not in path.read_text()
        restored = await Context.from_dict(flow, saved).store.get_state()
        assert restored.results[0].summary.screens[0][1] == prefix + "***" + " tail" * 80
    else:
        # The planner renders only the latest failure. Render the earlier failure
        # with the registry after the second child has registered the value.
        state = await handler.ctx.store.get_state()
        ctx = Context(flow)
        await ctx.store.set_state(
            state.model_copy(update={"results": state.results[:1], "current": None})
        )
        llm = Replies()
        memory = build_planner_memory(MemoryConfig(), state)
        try:
            await LlamaIndexPlanner(llm).bind(None, None, redact).plan(ctx, memory)
        finally:
            await memory.aclose()
        messages, _ = llm._requests[0]
        prompt = messages[1].content
        assert prefix + "fixt" not in prompt
        assert f"— {prefix} ×1" in prompt


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", ["checkpoint", "prompt"])
async def test_failure_summary_keeps_multiline_values_intact_until_redaction(
    make_workflow, phone, tmp_path, boundary
):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    secret = "synthetic-first-half\nsynthetic-second-half"
    before = UIState([], f"Sign in: {secret} unrelated tail", "", {}, 100, 100)
    after = UIState([], "Sign-in dialog closed", "", {}, 100, 100)
    phone._state_provider_override.get_state.side_effect = [before, before, after, after]
    redact = None

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            nonlocal redact
            redact = registry.redactor
            redact.add(secret)
            return Selection(
                calls=[ToolSelection(tool_id="tick", tool_name="tick", tool_kwargs={})]
            )

    llm = Replies()
    llm._calls = [
        ("set_plan", {"subgoals": ["Find price"]}),
        ("complete", {"success": False, "message": "Price unavailable"}),
    ]
    flow = await make_workflow(LlamaIndexPlanner(llm), Selector())
    handler = flow.run(goal="Find price", max_steps=3)
    assert (await handler).reason == "complete"
    if boundary == "checkpoint":
        assert redact is not None
        path = tmp_path / "context.json"
        saved = await save_context(path, handler.ctx, redact)
        assert "synthetic-first-half" not in path.read_text()
        state = await Context.from_dict(flow, saved).store.get_state()
        assert state.results[0].summary.screens[0][1] == "Sign in: *** unrelated tail"
    else:
        messages, _ = llm._requests[-1]
        prompt = messages[1].content
        assert "synthetic-first-half" not in prompt
        assert "— Sign in:  ×1" in prompt


@pytest.mark.anyio
async def test_child_budget_allows_replan_but_total_budget_prevents_another_child(make_workflow):
    planner = FakePlanner(Plan(("First",)), Plan(("Second",)), Plan(("Never starts",)))
    selector = FakeSelector(
        ToolSelection(tool_id="tick", tool_name="tick", tool_kwargs={}), complete()
    )
    config = AgentConfig(planning={"subgoal_steps": 1})
    flow = await make_workflow(planner, selector, config=config)
    handler = flow.run(goal="Recover", max_steps=5)
    events = [event async for event in handler.stream_events()]
    assert (await handler).reason == "max_steps"
    assert [event.steps for event in events if isinstance(event, Finished)] == [5]
    assert [event.step for event in events if isinstance(event, Observed)] == [0, 1, 2, 2, 4]
    assert planner.states[1].results[0].reason == "max_steps"
    assert selector.goals == ["First", "Second"]


@pytest.mark.anyio
async def test_child_cannot_consume_the_planners_final_step(make_workflow):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    llm = Replies()
    llm._calls = [
        ("set_plan", {"subgoals": ["Only child"]}),
        ("complete", {"success": False, "message": "The sign-in dialog blocked the price."}),
    ]
    tick = ToolSelection(tool_id="tick", tool_name="tick", tool_kwargs={})
    selector = FakeSelector(tick, tick)
    flow = await make_workflow(
        LlamaIndexPlanner(llm), selector, config=AgentConfig(planning={"subgoal_steps": 10})
    )
    handler = flow.run(goal="Check", max_steps=3)
    outcome = await handler
    assert not outcome.success and outcome.reason == "complete"
    assert outcome.message == "The sign-in dialog blocked the price."
    state = await handler.ctx.store.get_state()
    assert (state.planner_turns, state.executor_steps) == (2, 1)
    assert state.results[0].reason == "max_steps"
    assert selector.goals == ["Only child"]
    messages, tools = llm._requests[-1]
    assert [tool.metadata.get_name() for tool in tools] == ["complete"]
    assert "final planner turn" in messages[-1].content


@pytest.mark.anyio
@pytest.mark.parametrize("valid_final_reply", [True, False])
async def test_invalid_first_reply_preserves_final_complete_only_step(
    make_workflow, valid_final_reply
):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    llm = Replies()
    invalid = ("set_plan", {"subgoals": ["No room for a child"]})
    blocker = "Sign-in required"
    llm._calls = [
        invalid,
        ("complete", {"success": False, "message": blocker}) if valid_final_reply else invalid,
    ]
    config = AgentConfig(decision={"max_response_repair_attempts": 1})
    flow = await make_workflow(LlamaIndexPlanner(llm, config), FakeSelector(), config=config)
    handler = flow.run(goal="Find price", max_steps=2)
    steps = []
    async for event in handler.stream_events():
        match event:
            case Finished(steps=count):
                steps.append(count)
    result = await handler

    assert steps == [2]
    assert (result.reason, result.message) == (
        ("complete", blocker) if valid_final_reply else ("max_steps", "Step limit reached")
    )
    assert [[tool.metadata.get_name() for tool in tools] for _, tools in llm._requests] == [
        ["complete"],
        ["complete"],
    ]


@pytest.mark.anyio
async def test_custom_planner_cannot_finish_last_turn_with_blank_failure(make_workflow):
    flow = await make_workflow(FakePlanner(Finish(False, " \t\n")), FakeSelector())
    result = await flow.run(goal="Find price", max_steps=1)

    assert not result.success
    assert (result.reason, result.message) == ("max_steps", "Step limit reached")


@pytest.mark.anyio
async def test_planning_console_reports_subgoal_turn_and_final_budgets(phone, caplog):
    from mobilerun_agents.agent.operators import PlanningOperator
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    config = MobileConfig()
    config.analytics.enabled = False
    config.tracing.enabled = False
    config.checkpoints.enabled = False
    operator = PlanningOperator(
        FakePlanner(Plan(("Search eBay",)), Plan(("Try again",)), Finish(False, "Price hidden")),
        FakeSelector(complete(), complete()),
    )
    with ConfigLoader.override(config):
        result = await operator.execute(phone, "Find the price", max_steps=60)

    assert not result.success and result.answer == "Price hidden" and result.steps == 5
    assert "→ subgoal (step 1/60): Search eBay" in caplog.messages
    assert "→ subgoal (step 3/60): Try again" in caplog.messages
    assert "✗ Price hidden (5/60 steps · 3 planner, 2 actions)" in caplog.messages


@pytest.mark.anyio
@pytest.mark.parametrize("last_reply", ["failure", "invalid-tool", "invalid-output"])
async def test_last_planner_turn_reports_the_blocker_or_falls_back(make_workflow, last_reply):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    class FinalReply(Replies):
        async def achat(self, messages, tools, tool_required, **kwargs):
            if [tool.metadata.get_name() for tool in tools] == ["complete"]:
                assert "Sign-in required" in messages[1].content
                reply = {
                    "failure": (
                        "complete",
                        {"success": False, "message": "Cannot read the price: sign-in required."},
                    ),
                    "invalid-tool": ("set_plan", {"subgoals": ["Retry"]}),
                    "invalid-output": ("complete", {"success": True, "message": "Found it"}),
                }[last_reply]
                self._calls[len(self._requests)] = reply
            return await super().achat(messages, tools, tool_required, **kwargs)

    llm = FinalReply()
    llm._calls = [("set_plan", {"subgoals": ["Read price"]})] * 3
    config = AgentConfig(
        decision={"max_response_repair_attempts": 1},
        max_output_validation_attempts=1,
    )
    selector = FakeSelector(*(complete("Sign-in required", False) for _ in range(3)))
    flow = await make_workflow(
        LlamaIndexPlanner(llm, config), selector, config=config, output=Output
    )
    result = await flow.run(goal="Find the price", max_steps=5)

    assert not result.success and result.output is None
    assert (result.reason, result.message) == (
        ("complete", "Cannot read the price: sign-in required.")
        if last_reply == "failure"
        else ("max_steps", "Step limit reached")
    )
    assert selector.goals == ["Read price", "Read price"]


@pytest.mark.anyio
async def test_planner_can_finish_with_the_last_shared_step(make_workflow):
    planner = FakePlanner(Plan(("Read value",)), Finish(True, "Found it"))
    flow = await make_workflow(planner, FakeSelector(complete("Read 12")))
    assert (await flow.run(goal="Find value", max_steps=3)).success
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
    schema, repair = planner.prompt_inputs[1]
    assert schema["properties"]["price"]["type"] == "integer"
    assert "output is invalid" in repair


@pytest.mark.anyio
@pytest.mark.parametrize("protect", [False, True])
async def test_parent_cancellation_stops_child_and_purges_newly_protected_image(
    make_workflow, protect
):
    started = asyncio.Event()
    ended = asyncio.Event()

    class Waiting(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            if protect:
                registry.protect(
                    ToolSelection(
                        tool_id="protect", tool_name="protect", tool_kwargs={"text": "fake-secret"}
                    )
                )
            started.set()
            try:
                await asyncio.Future()
            finally:
                ended.set()

    planner = FakePlanner(Plan(("Wait",)))
    flow = await make_workflow(planner, Waiting())
    handler = flow.run(goal="Cancel")
    await asyncio.wait_for(started.wait(), timeout=5)
    path = planner.states[0].current.screenshot.path
    assert path.is_file()
    await handler.cancel_run()
    with pytest.raises(WorkflowCancelledByUser):
        await handler
    await asyncio.wait_for(ended.wait(), timeout=5)
    assert path.exists() is not protect


@pytest.mark.anyio
@pytest.mark.parametrize("batch", ["parallel", "sequential", "complete_only"])
async def test_planner_refreshes_after_each_kind_of_terminal_batch(make_workflow, phone, batch):
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
    provider.get_state.side_effect = [before, before, after, after]

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
    assert state.current.ui.formatted_text == "After"
    assert state.actions[0].before.ui.formatted_text == "Before"
    assert provider.get_state.await_count == (4 if batch == "sequential" else 3)
    assert state.current.screenshot.path.read_bytes() == b"After"
    assert phone.screenshot.await_count == 2


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

    summary = _render_failure(planner.states[1].results[0], planner.redact)
    suffix = "" if position in {"crossing", "encoded_crossing"} else marker[:remaining]
    assert f"- {thought_prefix}{suffix}\n" in summary
    assert f" — {screen_prefix}{planner.redact.text(suffix)} ×1\n" in summary


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

    summary = _render_failure(planner.states[1].results[0], planner.redact)
    assert "fixture-a" not in planner.redact.text(summary)
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
        if not any(item is memory for item in used):
            return
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

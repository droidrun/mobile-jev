import asyncio
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import AsyncMock
from urllib.parse import quote_plus

import pytest
from llama_index.core.base.llms.types import ImageBlock, TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from llama_index.llms.anthropic import Anthropic
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from PIL import Image
from pydantic import PrivateAttr
from workflows import Context

from mobilerun_agents import LlamaIndexPlanner, Planner, PlanningOperator, Run, tool
from mobilerun_agents.agent.action_selector import DecisionSelector
from mobilerun_agents.agent.action_selector.base import ActionSelection, ActionSelector
from mobilerun_agents.agent.action_selector.decision_model.base import Decision
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.llama_index import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.diagnostics import Detail, private_values
from mobilerun_agents.agent.logs.redaction import run_sensitive_values
from mobilerun_agents.agent.planner import Finish, Plan, PlanningInput, SubgoalResult
from mobilerun_agents.agent.tools.base import complete as finish_tool
from mobilerun_agents.agent.workflows.agent import AgentWorkflow
from mobilerun_agents.agent.workflows.events import Invalid, Select, StepFinished, Subgoal
from mobilerun_agents.agent.workflows.planning import PlannerScreenshot, PlanningWorkflow
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import (
    AgentConfig,
    DecisionConfig,
    MobileConfig,
    ToolsConfig,
)
from tests.unit.agent.workflows.test_run import Dated, Output, complete, resources
from tests.unit.agent.workflows.test_run import anyio_backend as anyio_backend
from tests.unit.agent.workflows.test_run import block_network as block_network


class PlannerLLM(MockLLM):
    _replies = PrivateAttr()
    _turns = PrivateAttr(default_factory=list)

    def __init__(self, *replies):
        super().__init__()
        self._replies = iter(replies)

    async def achat(self, messages, tools, tool_required, **kwargs):
        self._turns.append((list(messages), tools))
        assert tool_required
        name, args = next(self._replies)
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                blocks=[
                    TextBlock(text="Review progress. " + "x" * 300 + "TRUNCATED"),
                    ToolCallBlock(tool_name=name, tool_kwargs=args, tool_call_id="call"),
                ],
            ),
            raw={"usage": {"input_tokens": 4, "output_tokens": 3}},
        )

    async def astream_chat(self, messages, **kwargs):
        async def chunks():
            yield await self.achat(messages, **kwargs)

        return chunks()


class ScriptedPlanner(Planner):
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.turns = []

    async def plan(self, request):
        self.turns.append(request)
        return next(self.decisions)


class Executor(ActionSelector):
    def __init__(self, *calls):
        self.calls = iter(calls)
        self.turns = []

    async def select(self, run):
        self.turns.append((run.state, run.registry, run.output))
        run.state.goal = "corrupted"
        return ActionSelection(actions=[next(self.calls)], memory_update="counted")


@tool(sensitive_parameters=frozenset({"value"}))
async def authorize(run: Run, value: str) -> str:
    """Apply a fake authorization value."""
    return value


class Selections(ActionSelector):
    def __init__(self, *selections):
        self.selections = iter(selections)

    async def select(self, run):
        return next(self.selections)


@pytest.mark.anyio
async def test_queued_subgoal_is_raw_for_planner_but_masked_in_context():
    marker = "FAKE_QUEUE / REVIEW_9"
    queued = f"Later target {marker}"
    planner = ScriptedPlanner(Plan(("First",)), Plan(("Next", queued)), Finish(True, "done"))
    executor = Selections(
        ActionSelection(actions=[ToolCall("authorize", {"value": marker})]),
        ActionSelection(actions=[complete()]),
        ActionSelection(actions=[complete()]),
    )
    handler = PlanningWorkflow(planner, executor).run(**await resources(), tools=[authorize])
    assert (await handler).success
    assert planner.turns[-1].remaining_plan == (queued,)
    assert (await handler.ctx.store.get_state()).remaining_plan == ["Later target ***"]
    assert marker not in json.dumps(handler.ctx.to_dict())


@pytest.mark.anyio
async def test_active_subgoal_is_raw_for_executor_but_masked_in_both_contexts(monkeypatch):
    marker = "FAKE_ACTIVE_GOAL_876234"
    goal = f"Use {marker}"
    entered = asyncio.Event()
    resume = asyncio.Event()
    children = []
    observed = []
    original_run = AgentWorkflow.run

    def record_child(self, *args, **kwargs):
        handler = original_run(self, *args, **kwargs)
        children.append(handler)
        return handler

    class Executor(ActionSelector):
        async def select(self, run):
            if run.state.goal == "First" and not run.state.actions:
                return ActionSelection(actions=[ToolCall("authorize", {"value": marker})])
            if run.state.goal != "First":
                observed.append((run.state.goal, run.snapshot.goal))
                entered.set()
                await resume.wait()
            return ActionSelection(actions=[complete()])

    monkeypatch.setattr(AgentWorkflow, "run", record_child)
    planner = ScriptedPlanner(Plan(("First",)), Plan((goal,)), Finish(True, "done"))
    handler = PlanningWorkflow(planner, Executor()).run(**await resources(), tools=[authorize])
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        active = [json.dumps(child.ctx.to_dict()) for child in (handler, children[1])]
    finally:
        resume.set()
    events = [event async for event in handler.stream_events()]
    assert (await handler).success
    completed = [json.dumps(child.ctx.to_dict()) for child in (handler, *children)]
    assert observed == [(goal, goal)] and planner.turns[-1].results[-1].goal == goal
    assert [event.text for event in events if isinstance(event, Subgoal)] == ["First", "Use ***"]
    assert marker not in repr(events)
    assert all(marker not in snapshot for snapshot in [*active, *completed])
    assert (await children[1].ctx.store.get_state()).goal == "Use ***"


@pytest.mark.anyio
async def test_child_notes_are_raw_for_planner_but_masked_on_output():
    marker = "FAKE_NOTE / REVIEW_9"
    note = f"Keep {marker}"
    planner = ScriptedPlanner(Plan(("First",)), Finish(True, "done"))
    executor = Selections(
        ActionSelection(actions=[ToolCall("authorize", {"value": marker})], memory_update=note),
        ActionSelection(actions=[complete()]),
    )
    token = private_values.set(set())
    try:
        handler = PlanningWorkflow(planner, executor).run(**await resources(), tools=[authorize])
        events = [event async for event in handler.stream_events()]
        assert (await handler).success
    finally:
        private_values.reset(token)
    assert planner.turns[-1].notes == (note,)
    assert (await handler.ctx.store.get_state()).notes == ["Keep ***"]
    assert marker not in repr(events) and marker not in json.dumps(handler.ctx.to_dict())


@pytest.mark.anyio
async def test_planner_reasoning_masks_encoded_sensitive_values_before_clipping(caplog):
    marker = "FAKE private / marker plus tail REVIEW 9"
    encoded = quote_plus(marker)

    class EchoPlanner(PlannerLLM):
        async def achat(self, messages, **kwargs):
            response = await super().achat(messages, **kwargs)
            if len(self._turns) == 2:
                response.message.blocks[0] = TextBlock(text="x" * 264 + encoded)
            return response

    planner = EchoPlanner(
        ("set_plan", {"subgoals": ["First"]}), ("complete", {"success": True, "message": "done"})
    )
    executor = Selections(
        ActionSelection(actions=[ToolCall("authorize", {"value": marker})]),
        ActionSelection(actions=[complete()]),
    )
    token = private_values.set(set())
    try:
        handler = PlanningWorkflow(LlamaIndexPlanner(planner), executor).run(
            **await resources(), tools=[authorize]
        )
        events = [event async for event in handler.stream_events()]
        assert (await handler).success
        caplog.set_level("DEBUG")
        console = ConsoleLog()
        for event in events:
            console.on_event(event)
    finally:
        private_values.reset(token)
    details = [event.text for event in events if isinstance(event, Detail)]
    assert any(text.endswith("x" * 264 + "***") for text in details)
    assert not any(encoded[:24] in text for text in details)
    assert "reasoning:" in caplog.text and encoded[:24] not in caplog.text


@pytest.mark.anyio
async def test_failure_summary_keeps_blocker_from_repaired_attempt():
    blocker = "Sign-in dialog hides prices."
    planner = ScriptedPlanner(Plan(("Read prices",)), Finish(False, "incomplete"))
    executor = Selections(
        ActionSelection(actions=[], thought=blocker),
        ActionSelection(actions=[ToolCall("wait")], thought="Try waiting"),
    )
    handler = PlanningWorkflow(planner, executor).run(
        **await resources(),
        config=AgentConfig(planning={"subgoal_steps": 1}),
        tool_config=ToolsConfig(wait_duration=0.001),
    )
    await handler
    summary = planner.turns[-1].results[-1].failure
    assert summary.reason == "max_steps"
    assert blocker in summary.blockers


@pytest.mark.anyio
@pytest.mark.parametrize(("limit", "reason", "steps"), [(5, "max_steps", 5), (12, "cycle", 11)])
async def test_failed_child_reports_loop_evidence_and_planner_changes_approach(
    limit, reason, steps
):
    secret_parts = ("FAKE_SITE_VALUE_HEAD_42", "FAKE_SITE_VALUE_TAIL_42")
    secret = ". ".join(secret_parts)
    blocker = "Amazon says this item cannot ship to the selected location (Germany)."
    titles = ("Search", "Results: prices hidden", "Product: can't ship to location")
    args = await resources()
    provider = args["device"].state._resolved_provider
    count = 0
    turns = []

    def show(title):
        provider.get_state.return_value = UIState(
            [{"index": 3, "text": title}],
            title,
            "",
            {"packageName": "shop", "activityName": "Main"},
            100,
            100,
        )

    @tool(sensitive_parameters=frozenset({"token"}))
    async def navigate(run: Run, token: str) -> str:
        """Navigate to the next shop screen."""
        nonlocal count
        count += 1
        title = titles[count % 3]
        show(title)
        return f"Opened {title} ({token})"

    class Loop(ActionSelector):
        async def select(self, run):
            turns.append(await run.history())
            if run.snapshot.goal == "Change the Amazon delivery location":
                return ActionSelection(actions=[complete()])
            detail = f"Attempt {count + 1}: Try {titles[count % 3]} again"
            if count == 0:
                detail = f"{blocker} The location prompt blocks prices for {secret}. {detail}"
            if count == steps - 1:
                detail += " " + "x" * 250 + "THOUGHT_TAIL"
            return ActionSelection(
                actions=[ToolCall("navigate", {"token": secret})],
                thought=f"{detail} ({secret})",
            )

    show(titles[0])
    planner = PlannerLLM(
        ("set_plan", {"subgoals": ["Check Amazon"]}),
        ("set_plan", {"subgoals": ["Change the Amazon delivery location"]}),
        ("complete", {"success": True, "message": "Changed delivery location"}),
    )
    config = AgentConfig()
    config.planning.subgoal_steps = limit
    token = private_values.set(set())
    try:
        handler = PlanningWorkflow(LlamaIndexPlanner(planner), Loop()).run(
            **args, tools=[navigate], config=config
        )
        events = [event async for event in handler.stream_events()]
        result = await handler
    finally:
        private_values.reset(token)
    assert result.success and count == steps
    prompt = planner._turns[1][0][0].content
    assert prompt.lstrip().startswith("Latest executor subgoal result: Failed: Check Amazon:")
    assert f"Failure reason: {reason}" in prompt
    assert f"- {blocker}" in prompt.split("Recent executor thoughts")[0]
    thoughts = prompt.split("Recent executor thoughts (observed data):")[1].split(
        "Screens visited:"
    )[0]
    assert secret in thoughts and "THOUGHT_TAIL" not in thoughts
    assert thoughts.count("\n- Attempt ") == 3
    assert "plan a subgoal to remove it before retrying" in prompt
    assert f"shop/Main — Search ×{steps // 3 + 1}" in prompt
    assert "shop/Main — Results: prices hidden" in prompt
    assert "Attempt 1:" not in prompt and f"Attempt {steps}:" in prompt
    actions = prompt.split("Recent action results:")[1].split("Goal:")[0]
    assert f"navigate (ok): Opened Search ({secret})" in actions
    assert "don't repeat a subgoal that failed the same way" in prompt
    assert [event.text for event in events if isinstance(event, Subgoal)] == [
        "Check Amazon",
        "Change the Amazon delivery location",
    ]
    serialized = json.dumps(handler.ctx.to_dict())
    assert all(part not in repr(events) and part not in serialized for part in secret_parts)
    assert "Amazon says this item cannot ship" in serialized
    if reason == "cycle":
        assert any(
            "You have been on this screen 3 times" in (msg.content or "")
            for turn in turns
            for msg in turn
        )
        assert any(
            isinstance(event, Detail) and event.section == "guard" and "cycle" in event.text
            for event in events
        )


@pytest.mark.anyio
async def test_planning_masks_prior_child_secrets_on_stream_and_disk(tmp_path, monkeypatch):
    from mobilerun_agents.agent.operators._consumers import RunConsumers

    marker = "FAKE_TRACKED_PLANNING_VALUE_9347"
    args = await resources(Output)
    args["goal"] = f"Verify {marker}"
    device = args["device"]
    device._supported_operations = frozenset({"input_text", "open_url"})
    provider = device.state._resolved_provider
    provider.get_state.return_value.phone_state.update(focusedElement={"index": 1}, isEditable=True)
    await DictCredentials({"shop": marker}).bind(device)

    async def type_and_echo(text, *args, **kwargs):
        assert text == marker
        provider.get_state.return_value = UIState(
            [{"index": 3, "text": marker, "className": "TextView"}], marker, "", {}, 100, 100
        )
        return True

    async def open_url(url: str) -> None:
        pass

    device._input_text_raw = type_and_echo
    device.open_url = open_url
    public = ToolCall("open_url", {"url": "https://example.invalid/"})
    private = ToolCall("vault__type_field", {"package": None, "field": "password", "name": "shop"})
    echo = ToolCall("open_url", {"url": f"https://example.invalid/{marker}"})
    planner = PlannerLLM(
        ("set_plan", {"subgoals": ["one"]}),
        ("set_plan", {"subgoals": [f"two {marker}"]}),
        ("complete", {"success": True, "message": f"done {marker}", "output": {"count": 2}}),
    )
    config = MobileConfig()
    config.analytics.enabled = config.tracing.enabled = False
    config.checkpoints.folder = tmp_path / "runs"
    monkeypatch.setattr(
        "mobilerun_agents.agent.operators._consumers.get_macro_settings",
        lambda: SimpleNamespace(macro_directory=tmp_path / "macros"),
    )
    child_finish = ToolCall("complete", {"success": True, "message": f"verified {marker}"})

    class ObservingExecutor(Executor):
        async def select(self, run):
            assert await run.screenshot() == b"screen"
            return await super().select(run)

    executor = ObservingExecutor(public, private, complete(), echo, public, child_finish)
    token = private_values.set(set())
    try:
        async with RunConsumers(config, "test-run", 40) as consumers:
            on_event = consumers.on_event
            events = []

            async def capture(event):
                events.append(event)
                await on_event(event)

            consumers.on_event = capture
            handler = PlanningWorkflow(LlamaIndexPlanner(planner), executor).run(**args)
            result, folder = await consumers.consume(handler)
    finally:
        private_values.reset(token)
    assert result.answer == "done ***" and result.output == Output(count=2)
    assert executor.turns[3][0].goal == f"two {marker}"
    last_prompt = planner._turns[-1][0][0].content
    assert f"Goal: Verify {marker}" in last_prompt
    assert f"two {marker}" in last_prompt and f"verified {marker}" in last_prompt
    saved_subgoal = (await handler.ctx.store.get_state()).results[-1]
    assert saved_subgoal.goal == "two ***" and saved_subgoal.message == "verified ***"
    snapshots = [event.snapshot for event in events if isinstance(event, Select)]
    assert snapshots[3].goal == "two ***" and snapshots[3].ui_elements[0]["text"] == "***"
    screenshots = [event.screenshot for event in events if type(event) is Select]
    assert screenshots[0] == b"screen" and all(image is None for image in screenshots[2:])
    assert all(marker not in repr(event) for event in events)
    assert all(
        marker not in repr(event.pre_state) for event in events if isinstance(event, StepFinished)
    )
    assert all(marker not in path.read_text() for path in Path(folder).rglob("*.json"))
    macro_path = tmp_path / "macros/test-run/macro.json"
    assert marker not in macro_path.read_text()
    actions = json.loads(macro_path.read_text())["actions"]
    assert [a["action_type"] for a in actions] == ["open_url", "extension_call", "open_url"]
    assert actions[1]["arguments"]["name"] == "shop" and actions[-1]["pre_state"] is None


@pytest.mark.anyio
async def test_plain_text_reaches_planner_and_console(caplog):
    from mobilerun_agents.agent.operators._consumers import RunConsumers
    from mobilerun_agents.agent.workflows.events import Act

    text = "iPhone 15 Pro Max"
    args = await resources()
    device = args["device"]
    device._supported_operations = frozenset({"input_text", "set_clipboard"})
    device._input_text_raw = AsyncMock(return_value=True)
    device.set_clipboard = AsyncMock()
    planner = PlannerLLM(
        ("set_plan", {"subgoals": ["Search Amazon"]}),
        ("complete", {"success": True, "message": "done"}),
    )
    executor = Executor(
        ToolCall("input_text", {"text": text}),
        ToolCall("set_clipboard", {"text": text}),
        ToolCall("complete", {"success": True, "message": f"Searched for {text}"}),
    )
    config = MobileConfig()
    config.analytics.enabled = config.tracing.enabled = config.checkpoints.enabled = False
    caplog.set_level("DEBUG")
    token = private_values.set(set())
    try:
        async with RunConsumers(config, "plain-text", 10) as consumers:
            handler = PlanningWorkflow(LlamaIndexPlanner(planner), executor).run(**args)
            events = [event async for event in handler.stream_events()]
            for event in events:
                await consumers.on_event(event)
    finally:
        private_values.reset(token)

    assert (await handler).success
    assert f"Searched for {text}" in planner._turns[1][0][0].content
    picks = [ev.selection.actions[0] for ev in events if type(ev) is Act]
    assert [p.parameters["text"] for p in picks if "text" in p.parameters] == [text, text]
    assert text in caplog.text


@pytest.mark.parametrize("module", [__name__, "__main__"])
def test_custom_llm_diagnostic_name(module):
    class CustomLLM(MockLLM):
        pass

    CustomLLM.__module__ = module
    llm = CustomLLM()
    assert (
        LlamaIndexLLMSelector(llm, vision="off").diagnostic_name
        == f"llm:CustomLLM/{llm.metadata.model_name}"
    )


@pytest.mark.anyio
async def test_executor_tool_receives_the_same_subgoal_run_as_selector():
    seen = []

    @tool
    async def inspect_goal(run: Run) -> str:
        """Record which run was injected."""
        seen.append((run, run.state.goal, run.output))
        return "ok"

    class Inspecting(ActionSelector):
        async def select(self, run):
            if seen:
                return ActionSelection(actions=[complete()])
            self.turn = run
            return ActionSelection(actions=[ToolCall("inspect_goal")])

    selector = Inspecting()
    planner = ScriptedPlanner(Plan(("subgoal",)), Finish(True, "done", {"count": 2}))
    result = await PlanningWorkflow(planner, selector).run(
        **await resources(Output), tools=[inspect_goal, finish_tool]
    )
    assert result.success and seen == [(selector.turn, "subgoal", None)]


@pytest.mark.anyio
async def test_planner_mapping_output_is_validated_as_requested_model():
    flow = PlanningWorkflow(ScriptedPlanner(Finish(True, "done", {"count": 2})), Executor())
    result = await flow.run(**await resources(Output))
    assert result.success and result.output == Output(count=2)


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", [None, {"day": "not a date"}])
async def test_planner_repairs_bad_output_using_output_budget(invalid):
    planner = PlannerLLM(
        ("complete", complete(invalid).parameters),
        ("complete", complete({"day": "2026-11-24"}).parameters),
    )
    flow = PlanningWorkflow(LlamaIndexPlanner(planner), Executor())
    config = AgentConfig(
        max_output_validation_attempts=2, decision=DecisionConfig(max_response_repair_attempts=1)
    )
    result = await flow.run(**await resources(Dated), config=config)
    assert result.success and result.output.model_dump(mode="json") == {"day": "2026-11-24"}
    retry = "\n".join(m.content or "" for m in planner._turns[1][0])
    assert "output" in retry and "invalid" in retry and "not a date" not in retry


@pytest.mark.anyio
async def test_planner_stops_when_output_budget_is_exhausted():
    planner = PlannerLLM(*[("complete", complete({"count": "2"}).parameters)] * 2)
    flow = PlanningWorkflow(LlamaIndexPlanner(planner), Executor())
    config = AgentConfig(
        max_output_validation_attempts=2, decision=DecisionConfig(max_response_repair_attempts=4)
    )
    result = await flow.run(**await resources(Output), config=config)
    assert not result.success and result.reason == "output_attempts" and result.steps == 0
    assert len(planner._turns) == 2


@pytest.mark.anyio
async def test_workflow_repairs_an_empty_custom_plan():
    planner = ScriptedPlanner(Plan(()), Plan(("subgoal",)), Finish(True, "done"))
    flow = PlanningWorkflow(planner, Executor(complete()))
    handler = flow.run(**await resources())
    events = [event async for event in handler.stream_events()]
    result = await handler
    assert result.success and result.steps == 1
    assert sum(type(event) is Invalid for event in events) == 1


@pytest.mark.anyio
async def test_planner_receives_screen_summary_without_screenshot():
    class SeeingPlanner(Planner):
        async def plan(self, request):
            assert request.ui == "screen"
            return Finish(True, "done")

    handler = PlanningWorkflow(SeeingPlanner(), Executor()).run(**await resources())
    turns = [event async for event in handler.stream_events() if isinstance(event, Select)]
    assert (await handler).success
    assert not turns


@pytest.mark.anyio
async def test_plan_returns_plan_from_required_tool_without_screenshot():
    run = PlanningInput("Count items", AgentConfig())
    planner = PlannerLLM(("set_plan", {"subgoals": [" Check items "]}))
    assert await LlamaIndexPlanner(planner).plan(run) == Plan(("Check items",))
    messages, tools = planner._turns[0]
    assert all(block.block_type == "text" for m in messages for block in m.blocks)
    assert [t.metadata.get_name() for t in tools] == ["set_plan", "complete"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "message,vision,observe,protected",
    [
        ("", True, True, False),
        ("", True, True, True),
        ("  ", True, True, False),
        ("Goal achieved.", True, True, False),
        ("Goal cannot be achieved with the available actions.", True, True, False),
        ("complete: subgoal done", True, True, False),
        ("complete: report failure (blocked or impossible)", True, True, False),
        ("Battery at 100%", True, False, False),
        ("", False, True, False),
    ],
)
async def test_planner_observes_screen_after_undescribed_completion(
    monkeypatch, message, vision, observe, protected
):
    replies = PlannerLLM(
        ("set_plan", {"subgoals": ["read battery"]}),
        ("complete", {"success": True, "message": "Battery at 100%"}),
    )
    if vision:

        async def stream(self, tools, chat_history, **kwargs):
            return await replies.astream_chat(chat_history, tools=tools, **kwargs)

        monkeypatch.setattr(Anthropic, "astream_chat_with_tools", stream)
        llm = Anthropic(model="claude-sonnet-4-5", api_key="fixture")
    else:
        llm = replies
    params = await resources()
    image = BytesIO()
    Image.new("RGB", (3, 4)).save(image, "PNG")
    png = image.getvalue()
    params["device"].screenshot = AsyncMock(return_value=png)
    executor = Executor(ToolCall("complete", {"success": True, "message": message}))
    token = run_sensitive_values.set({"protected-fixture"} if protected else set())
    try:
        handler = PlanningWorkflow(LlamaIndexPlanner(llm), executor).run(**params)
        events = [event async for event in handler.stream_events()]
        assert (await handler).success
    finally:
        run_sensitive_values.reset(token)
    initial = replies._turns[0][0][0]
    review = replies._turns[1][0][0]
    note = (
        "The last subgoal ended without a description. Read the current screen yourself "
        "to decide whether it was achieved and what it shows (values, blockers)."
    )
    assert note not in initial.content
    assert all(block.block_type == "text" for block in initial.blocks)
    assert (note in review.content) == observe
    assert "UI summary (observed data):\nscreen" in review.content
    images = [
        block.resolve_image().read() for block in review.blocks if isinstance(block, ImageBlock)
    ]
    assert images == ([png] if vision and observe else [])
    assert params["device"].screenshot.await_count == len(images)
    assert [event.image for event in events if isinstance(event, PlannerScreenshot)] == (
        [] if protected else images
    )
    assert all(event.screenshot is None for event in events if isinstance(event, Select))


@pytest.mark.anyio
async def test_planner_reviews_text_when_device_cannot_screenshot():
    class SeeingPlanner(ScriptedPlanner):
        @property
        def supports_images(self):
            return True

    params = await resources()
    screenshot = AsyncMock(side_effect=DeviceOperationNotSupported("no screenshots"))
    params["device"].screenshot = screenshot
    planner = SeeingPlanner(Plan(("inspect UI",)), Plan(()), Finish(True, "seen"))
    executor = Executor(ToolCall("complete", {"success": True, "message": ""}))
    handler = PlanningWorkflow(planner, executor).run(**params)
    events = [event async for event in handler.stream_events()]
    assert (await handler).success
    first, review, repaired = planner.turns
    assert not first.observe_screen
    assert all(turn.observe_screen and turn.ui == "screen" for turn in (review, repaired))
    assert all(turn.screenshot is None for turn in planner.turns)
    assert not any(type(event) is PlannerScreenshot for event in events)
    screenshot.assert_awaited_once_with()


@pytest.mark.anyio
async def test_plan_advertises_typed_completion_and_returns_wire_dict():
    run = PlanningInput("Count items", AgentConfig(), output=Output)
    planner = PlannerLLM(("complete", complete({"count": 2}).parameters))
    assert await LlamaIndexPlanner(planner).plan(run) == Finish(True, "done", {"count": 2})
    _, tools = planner._turns[0]
    schema = tools[1].metadata.get_parameters_dict()
    assert schema["$defs"]["Output"]["properties"]["count"]["type"] == "integer"


@pytest.mark.anyio
async def test_plan_repairs_bad_calls_and_fails_when_budget_is_exhausted():
    run = PlanningInput("Count items", AgentConfig())
    run.config.decision.max_response_repair_attempts = 2
    planner = PlannerLLM(("set_plan", '{"subgoals":['), ("set_plan", {"subgoals": [" "]}))
    with pytest.raises(InvalidResponse):
        await LlamaIndexPlanner(planner).plan(run)
    assert len(planner._turns) == 2
    messages, _ = planner._turns[1]
    assert "previous response was invalid" in (messages[-1].content or "")


@pytest.mark.anyio
async def test_workflow_ends_cleanly_when_planner_never_calls_a_known_tool():
    planner = PlannerLLM(("click", {"index": 3}))
    flow = PlanningWorkflow(LlamaIndexPlanner(planner), Executor())
    config = AgentConfig(decision=DecisionConfig(max_response_repair_attempts=1))
    result = await flow.run(**await resources(), config=config)
    assert not result.success and result.reason == "repair_attempts" and result.steps == 0


@pytest.mark.anyio
async def test_subgoals_return_to_planner_before_typed_completion():
    planner = PlannerLLM(
        ("set_plan", {"subgoals": ["first", "second"]}),
        ("set_plan", {"subgoals": ["second"]}),
        ("complete", complete({"count": 2}).parameters),
    )
    executor = Executor(complete(), ToolCall("complete", {"success": False, "message": "blocked"}))
    flow = PlanningWorkflow(planner=LlamaIndexPlanner(planner), executor=executor)
    token = private_values.set(set())
    try:
        handler = flow.run(**await resources(Output))
        events = [e async for e in handler.stream_events()]
        result = await handler
    finally:
        private_values.reset(token)
    assert result.output == Output(count=2) and result.steps == 2
    assert [e.text for e in events if isinstance(e, Subgoal)] == ["first", "second"]
    assert [s.goal for s, _, _ in executor.turns] == ["first", "second"]
    assert all(
        "tap_element" in c.by_name and "set_plan" not in c.by_name and out is None
        for _, c, out in executor.turns
    )
    review = planner._turns[-1][0][0].content
    assert "first: done" in review and "second: blocked" in review and "counted" in review
    assert "Failure reason:" not in review
    assert review.index("second: blocked") < review.index("first: done")
    details = [e.text for e in events if isinstance(e, Detail) and e.section == "planning"]
    assert any("first" in text and "second" in text for text in details)
    assert any("Review progress." in text for text in details)
    assert not any("TRUNCATED" in text for text in details)
    state = await handler.ctx.store.get_state()
    assert state.goal == "count items" and state.remaining_plan == [] and state.planner_turns == 3
    assert any("turn 3 · remaining plan 0 · last result ✗" == text for text in details)


@pytest.mark.anyio
async def test_planner_prompt_lists_only_executor_subgoal_results():
    run = PlanningInput(
        "Check the battery",
        AgentConfig(),
        results=(
            SubgoalResult(goal="Read battery", success=True, message="Battery at 100%", steps=1),
        ),
    )
    model = PlannerLLM(("complete", {"success": True, "message": "Battery at 100%"}))
    await LlamaIndexPlanner(model).plan(run)
    prompt = model._turns[0][0][0].content
    assert "Latest executor subgoal result: Succeeded: Read battery: Battery at 100%" in prompt


@pytest.mark.anyio
async def test_battery_goal_completes_after_executor_reads_values():
    answer = "Battery is at 100% and fully charged. Battery Saver is off."

    class BatteryPlanner(PlannerLLM):
        async def achat(self, messages, **kwargs):
            if self._turns:
                prompt = messages[0].content
                assert answer in prompt
                assert "call `complete` now" in prompt
                assert "verify, confirm, summarize, or report" in prompt
            return await super().achat(messages, **kwargs)

    class BatteryExecutor(PlannerLLM):
        async def achat(self, messages, **kwargs):
            if len(self._turns) == 3:
                assert "put the values you read into the message" in messages[0].content
            return await super().achat(messages, **kwargs)

    plan = BatteryPlanner(
        ("set_plan", {"subgoals": ["Open Settings and read battery status"]}),
        ("complete", {"success": True, "message": answer}),
    )
    execute = BatteryExecutor(
        ("tap_element", {"index": 3}),
        ("tap_element", {"index": 4}),
        ("tap_element", {"index": 5}),
        ("complete", {"success": True, "message": answer}),
    )
    args = await resources()
    args["goal"] = "Open Settings and check the battery status"
    provider = args["device"].state._resolved_provider
    provider.requires_active_contract_for_coords = False
    screens = iter([(4, "Battery"), (5, "Battery Saver"), (6, answer)])

    async def next_screen(x, y):
        index, text = next(screens)
        provider.get_state.return_value = UIState(
            [{"index": index, "text": text, "bounds": "0,0,20,40"}], "screen", "", {}, 100, 100
        )

    args["device"].tapped.side_effect = next_screen
    flow = PlanningWorkflow(
        LlamaIndexPlanner(plan),
        LlamaIndexLLMSelector(execute, vision="off", native_tool_api="responses"),
    )
    result = await flow.run(**args)
    assert result.success and result.answer == answer
    assert len(plan._turns) == 2 and len(execute._turns) == result.steps == 4


@pytest.mark.anyio
async def test_subgoal_limit_returns_failure_and_replans_with_continuous_events():
    planner = PlannerLLM(
        ("set_plan", {"subgoals": ["first", "remaining"]}),
        ("set_plan", {"subgoals": ["replacement"]}),
        ("complete", {"success": True, "message": "done"}),
    )
    executor = Executor(ToolCall("tap_element", {"index": 3}), complete())
    handler = PlanningWorkflow(LlamaIndexPlanner(planner), executor).run(
        **await resources(), config=AgentConfig(planning={"subgoal_steps": 1})
    )
    events = [e async for e in handler.stream_events()]
    assert (await handler).success
    assert [e.text for e in events if isinstance(e, Subgoal)] == ["first", "replacement"]
    assert [e.action.step for e in events if isinstance(e, StepFinished)] == [1, 2]
    assert [e.snapshot.step for e in events if isinstance(e, Select)] == [1, 2]
    prompt = planner._turns[1][0][0].content
    assert "Failed: first: step limit reached" in prompt and '["remaining"]' in prompt


@pytest.mark.anyio
async def test_total_executor_budget_limits_later_child():
    planner = ScriptedPlanner(Plan(("one",)), Plan(("two",)))
    executor = Executor(complete(), ToolCall("tap_element", {"index": 3}))
    result = await PlanningWorkflow(planner, executor).run(**await resources(), max_steps=2)
    assert not result.success and result.reason == "max_steps" and result.steps == 2
    assert len(planner.turns) == 2


@pytest.mark.anyio
async def test_planner_turn_limit_is_independent_from_executor_steps():
    planner = ScriptedPlanner(Plan(("one",)))
    result = await PlanningWorkflow(planner, Executor(complete())).run(
        **await resources(), config=AgentConfig(planning={"max_planner_turns": 1})
    )
    assert not result.success and result.reason == "max_planner_turns" and result.steps == 1


@pytest.mark.anyio
@pytest.mark.parametrize("timeout", [None, 0.5])
async def test_parent_cancellation_or_timeout_stops_child(timeout):
    from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError

    started, ended = asyncio.Event(), asyncio.Event()

    class Waiting(ActionSelector):
        async def select(self, run):
            started.set()
            try:
                await asyncio.Future()
            finally:
                ended.set()

    handler = PlanningWorkflow(ScriptedPlanner(Plan(("one",))), Waiting(), timeout=timeout).run(
        **await resources()
    )
    await asyncio.wait_for(started.wait(), 2)
    if timeout is None:
        await handler.cancel_run()
    with pytest.raises(WorkflowCancelledByUser if timeout is None else WorkflowTimeoutError):
        await handler
    await asyncio.wait_for(ended.wait(), 2)


@pytest.mark.anyio
async def test_remaining_plan_and_result_are_checkpointed_at_total_limit():
    flow = PlanningWorkflow(ScriptedPlanner(Plan(("a", "b"))), Executor(complete()))
    stopped = flow.run(**await resources(), max_steps=1)
    assert (await stopped).reason == "max_steps"
    restored = await Context.from_dict(flow, stopped.ctx.to_dict()).store.get_state()
    assert restored.remaining_plan == ["b"] and restored.results[0].goal == "a"


@pytest.mark.anyio
async def test_jev_executes_subgoal_with_an_llm_planner():
    from tests.unit.agent.action_selector.test_selectors import FakeModel

    class JevModel(FakeModel):
        async def decide(self, request):
            assert request.goal == "Inspect the row"
            return Decision(
                option_index=next(
                    i
                    for i, option in enumerate(request.options)
                    if option.action.name == "complete" and option.action.parameters["success"]
                ),
                message="row checked",
            )

    planner = ScriptedPlanner(
        Plan(("Inspect the row",)),
        Finish(True, "all done", {"count": 1}),
    )
    args = await resources(Output)
    args["device"].state._resolved_provider.requires_active_contract_for_coords = False
    result = await PlanningWorkflow(planner, DecisionSelector(JevModel(), vision="off")).run(**args)
    assert result.success and result.answer == "all done" and result.output == Output(count=1)
    assert planner.turns[-1].results[-1].message == "row checked"


def test_agent_workflow_has_no_planning_hooks():
    import inspect

    assert not issubclass(PlanningWorkflow, AgentWorkflow)
    source = inspect.getsource(inspect.getmodule(AgentWorkflow))
    assert "planner" not in source and "planning" not in source
    assert not any(
        name in source for name in ("subgoal", "_control_feedback", "_apply_feedback", "_make_run")
    )


def test_planner_parameter_is_annotated_as_planner():
    assert get_type_hints(PlanningOperator.__init__)["planner"] == Planner | None


@pytest.mark.anyio
async def test_planner_and_executor_resolve_once_from_default_setup(monkeypatch):
    from mobilerun_agents.agent.action_selector.loader import SetupSpec
    from mobilerun_agents.agent.operators import planning

    planner, executor = ScriptedPlanner(Finish(True, "done")), Executor()
    resolve = AsyncMock(return_value=SetupSpec(planner=planner, executor=executor))
    monkeypatch.setattr(planning, "default_setup", resolve)
    args = await resources()
    config = MobileConfig()
    config.analytics.enabled = config.checkpoints.enabled = False
    with ConfigLoader.override(config):
        result = await PlanningOperator().execute(args["device"], args["goal"])
    assert result.success and len(planner.turns) == 1 and not executor.turns
    resolve.assert_awaited_once_with()

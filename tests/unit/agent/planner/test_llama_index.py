import re

import pytest
from llama_index.core.base.llms.types import ImageBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from llama_index.core.memory import InsertMethod
from llama_index.core.utils import get_tokenizer
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from pydantic import PrivateAttr
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.action_selector.llm.models import Capabilities
from mobilerun_agents.agent.memory.blocks.findings import FindingsBlock
from mobilerun_agents.agent.memory.blocks.subgoals import LastSubgoalBlock, SubgoalHistoryBlock
from mobilerun_agents.agent.memory.memory import build_planner_memory
from mobilerun_agents.agent.planner import Finish, LlamaIndexPlanner, Plan
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import PlanningState, SubgoalResult
from mobilerun_agents.config.user_config import MobileConfig


class PlanningProbe(Workflow):
    @step
    async def start(self, ctx: Context[PlanningState], ev: StartEvent) -> StopEvent:
        return StopEvent()


class Replies(MockLLM):
    _requests = PrivateAttr(default_factory=list)
    _calls = PrivateAttr(
        default_factory=lambda: [
            ("set_plan", {"subgoals": [" "]}),
            ("set_plan", {"subgoals": [" Read battery "]}),
            ("complete", {"success": True, "message": "Battery at 100%"}),
        ]
    )

    async def achat(self, messages, tools, tool_required, **kwargs):
        assert tool_required
        self._requests.append((messages, tools))
        name, args = self._calls[len(self._requests) - 1]
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                blocks=[ToolCallBlock(tool_name=name, tool_kwargs=args, tool_call_id="reply")],
            )
        )

    async def astream_chat(self, messages, **kwargs):
        async def chunks():
            yield await self.achat(messages, **kwargs)

        return chunks()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def memory():
    memory = build_planner_memory(MobileConfig(), PlanningState(), "android")
    yield memory
    await memory.aclose()


@pytest.mark.anyio
async def test_llama_index_requires_a_valid_planner_tool_call(memory):
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(PlanningState(goal="Check the battery"))
    llm = Replies()
    planner = LlamaIndexPlanner(llm)

    plan = await planner.plan(ctx, memory)
    assert plan == Plan(("Read battery",))
    assert plan.findings == ()
    assert await planner.plan(ctx, memory) == Finish(True, "Battery at 100%")
    messages, tools = llm._requests[1]
    assert [tool.metadata.get_name() for tool in tools] == ["set_plan", "complete"]
    assert [message.role.value for message in messages] == ["system", "user", "user"]
    assert "Goal: Check the battery" in messages[1].content
    assert "previous response was invalid" in messages[-1].content


@pytest.mark.anyio
async def test_set_plan_carries_findings(memory):
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(PlanningState(goal="Compare battery levels"))
    llm = Replies()
    llm._calls = [("set_plan", {"subgoals": ["Check tablet"], "findings": ["Phone battery: 100%"]})]

    plan = await LlamaIndexPlanner(llm).plan(ctx, memory)

    assert plan == Plan(("Check tablet",), findings=("Phone battery: 100%",))
    _, (set_plan, _) = llm._requests[0]
    schema = set_plan.metadata.get_parameters_dict()
    assert schema["properties"]["findings"]["type"] == "array"
    assert schema["properties"]["findings"]["items"] == {"type": "string"}
    assert "findings" not in schema["required"]


@pytest.mark.anyio
@pytest.mark.parametrize("remaining", [0, 1])
async def test_last_planner_turn_offers_only_complete_even_during_repair(remaining, memory):
    ctx = Context[PlanningState](PlanningProbe())
    llm = Replies()
    llm._calls = [
        ("set_plan", {"subgoals": ["Read battery"]}),
        ("set_plan", {"subgoals": ["Retry battery"]}),
        ("complete", {"success": False, "message": "Battery settings are locked."}),
    ]
    planner = LlamaIndexPlanner(llm)

    await ctx.store.set_state(
        PlanningState(goal="Read battery", planner_turns=1, executor_steps=2, max_steps=5)
    )
    assert await planner.plan(ctx, memory) == Plan(("Read battery",))
    await ctx.store.set_state(
        PlanningState(
            goal="Read battery", planner_turns=3 - remaining, executor_steps=2, max_steps=5
        )
    )
    assert await planner.plan(ctx, memory) == Finish(False, "Battery settings are locked.")

    earlier, *last = llm._requests
    assert [tool.metadata.get_name() for tool in earlier[1]] == ["set_plan", "complete"]
    normal_prompt = earlier[0][0].content
    repeated_failure = "Repeated failures on the same kind of step"
    observed_or_unknown = (
        'Separate what you observed from what you infer. If the cause is unclear, say "unknown".'
    )
    assert "final planner turn" not in normal_prompt
    assert repeated_failure in normal_prompt
    assert "regardless of executor type" in normal_prompt
    assert "Do not just reword the subgoal" in normal_prompt
    assert (
        "try an approach the executor can do or call `complete` with success=false" in normal_prompt
    )
    assert (
        "say which capability was missing, based on the executor setup and results" in normal_prompt
    )
    assert observed_or_unknown in normal_prompt
    assert len(last) == 2
    for messages, tools in last:
        assert [tool.metadata.get_name() for tool in tools] == ["complete"]
        assert "final planner turn" in messages[-1].content
        assert "success=false" in messages[-1].content
        assert "state what blocked it" in messages[-1].content
        assert observed_or_unknown in messages[0].content
        assert repeated_failure not in messages[0].content
        assert "`set_plan`" not in messages[0].content


@pytest.mark.anyio
async def test_last_planner_turn_repairs_blank_failure_message(memory):
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(PlanningState(goal="Read battery", planner_turns=1, max_steps=1))
    llm = Replies()
    llm._calls = [
        ("complete", {"success": False, "message": " \t\n"}),
        ("complete", {"success": False, "message": "Battery settings are locked."}),
    ]

    assert await LlamaIndexPlanner(llm).plan(ctx, memory) == Finish(
        False, "Battery settings are locked."
    )
    assert len(llm._requests) == 2
    messages, tools = llm._requests[-1]
    assert "previous response was invalid" in messages[-1].content
    assert [tool.metadata.get_name() for tool in tools] == ["complete"]


@pytest.mark.anyio
@pytest.mark.parametrize("vision", [True, False])
async def test_llm_gets_path_backed_screenshot_if_it_supports_vision(
    tmp_path, monkeypatch, vision, memory
):
    path = tmp_path / "screen.png"
    path.write_bytes(b"fake screenshot payload")
    monkeypatch.setattr(
        "mobilerun_agents.agent.planner.llama_index.adapter_capabilities",
        lambda llm: Capabilities(vision=vision),
    )
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(
        PlanningState(
            goal="Check the battery",
            current=StepSnapshot(
                ui=UIState([], "Battery 100%", "", {}, 100, 100), screenshot=ImageBlock(path=path)
            ),
            results=[
                SubgoalResult(
                    subgoal="Read battery", success=True, message="Done", reason="complete"
                )
            ],
        )
    )
    llm = Replies()
    await LlamaIndexPlanner(llm).plan(ctx, memory)
    messages, _ = llm._requests[0]
    assert (
        "Verify executor results against the current screen and screenshot" in messages[0].content
    )
    assert "Battery 100%" in messages[-1].content
    images = [block for block in messages[-1].blocks if isinstance(block, ImageBlock)]
    assert [image.path for image in images] == ([path] if vision else [])


@pytest.mark.anyio
async def test_planner_config_removes_findings_and_orders_remaining_blocks():
    state = PlanningState(goal="Check the battery", notes=["Measured 60% earlier"], executor="llm")
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(state)
    llm = Replies()
    llm._calls = [("complete", {"success": True, "message": "Done"})]
    config = MobileConfig(memory={"planner": {"blocks": ["goal", "setup"]}})
    memory = build_planner_memory(config, state, "android")
    try:
        await LlamaIndexPlanner(llm).plan(ctx, memory)
    finally:
        await memory.aclose()
    (system, memory_message, step), _ = llm._requests[0]
    assert [message.role.value for message in (system, memory_message, step)] == [
        "system",
        "user",
        "user",
    ]
    assert memory_message.content.index("Goal: Check the battery") < memory_message.content.index(
        "Executor: an LLM"
    )
    assert all(
        "Measured 60% earlier" not in message.content for message in (system, memory_message, step)
    )


@pytest.mark.anyio
@pytest.mark.parametrize("insert_method", list(InsertMethod))
async def test_planner_request_renders_default_memory_order(insert_method):
    state = PlanningState(
        goal="Compare battery levels",
        executor="llm",
        notes=["Phone battery: 100%"],
        results=[
            SubgoalResult(subgoal="Open settings", success=True, message="Done", reason="complete"),
            SubgoalResult(subgoal="Read battery", success=True, message="100%", reason="complete"),
        ],
        current=StepSnapshot(
            ui=UIState(
                [],
                "Battery 100%",
                "",
                {"packageName": "settings", "activityName": "Battery"},
                100,
                100,
            )
        ),
    )
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(state)
    llm = Replies()
    llm._calls = [("set_plan", {"subgoals": ["Read tablet"]})]
    memory = build_planner_memory(
        MobileConfig(memory={"insert_method": insert_method}), state, "android"
    )
    try:
        await LlamaIndexPlanner(llm).plan(ctx, memory)
    finally:
        await memory.aclose()

    messages, _ = llm._requests[0]
    expected_roles = (
        ["system", "user"] if insert_method is InsertMethod.SYSTEM else ["system", "user", "user"]
    )
    assert [message.role for message in messages] == expected_roles
    assert re.findall(r"^<(\w+)>$", messages[-2].content, re.M) == [
        "memory",
        "setup",
        "goal",
        "findings",
        "last_subgoal",
        "subgoal_history",
    ]
    assert sum(message.content.count("Phone battery: 100%") for message in messages) == 1
    assert "Battery 100%" in messages[-1].content


@pytest.mark.anyio
async def test_planner_blocks_render_latest_result_and_bounded_history():
    def result(subgoal, message="Done", **screen):
        return SubgoalResult(
            subgoal=subgoal, success=True, message=message, reason="complete", **screen
        )

    state = PlanningState(
        notes=["Battery is charged"],
        results=[
            result("Too old"),
            result("Oldest in range", "first line\nsecond line"),
            result("Recent"),
            result(
                "Latest",
                before_screen=("shop", "List"),
                after_screen=("shop", "Item"),
                screen_changed=True,
            ),
        ],
    )
    assert await FindingsBlock().render(state) == 'Findings and notes: ["Battery is charged"]'
    assert (await LastSubgoalBlock().render(state)).splitlines() == [
        "Latest executor subgoal result: Succeeded: Latest: Done",
        "Screen: shop/List → shop/Item (changed)",
        "Reason: complete",
    ]
    history = SubgoalHistoryBlock(history=2)
    content = await history.render(state)
    recent = "Subgoal results (older):\n- Succeeded: Recent: Done"
    assert content.splitlines() == [
        "Subgoal results (older):",
        "- Succeeded: Oldest in range: first line second line",
        "- Succeeded: Recent: Done",
    ]
    assert await history.atruncate(content, 1) == recent
    assert await history.atruncate(content, len(get_tokenizer()(content))) == ""


@pytest.mark.anyio
async def test_planner_first_turn_renders_pinned_memory_under_token_pressure():
    state = PlanningState(goal="Check battery")
    memory = build_planner_memory(MobileConfig(memory={"token_limit": 1}), state, "android")
    try:
        (message,) = await memory.aget(state=state)
        assert "Goal: Check battery" in message.content
        assert "<subgoal_history>" not in message.content
        assert await memory.memory_blocks[-1].atruncate([], 1) == ""
    finally:
        await memory.aclose()


@pytest.mark.anyio
async def test_planner_sends_device_context_only_in_step():
    state = PlanningState(
        goal="Check battery",
        results=[
            SubgoalResult(
                subgoal="Open settings",
                success=True,
                message="Done",
                reason="complete",
                before_screen=("settings", "Main"),
                after_screen=("settings", "Battery"),
            )
        ],
        current=StepSnapshot(
            ui=UIState([], "Battery screen", "", {}, 100, 100),
            device_context=[ContextItem(source="battery", kind="status", data={"level": 63})],
        ),
    )
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(state)
    llm = Replies()
    llm._calls = [("set_plan", {"subgoals": ["Read battery"]})]
    memory = build_planner_memory(MobileConfig(), state, "android")
    try:
        await LlamaIndexPlanner(llm).bind(None, None).plan(ctx, memory)
    finally:
        await memory.aclose()
    (system, memory_message, step), _ = llm._requests[0]
    assert "settings/Main → settings/Battery" in memory_message.content
    assert "<device_context>" in step.content and '"level": 63' in step.content
    assert "<device_context>" not in system.content + memory_message.content

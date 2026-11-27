import asyncio
import base64
import json

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from workflows import Context, Workflow, step
from workflows.context.serializers import JsonSerializer
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.checkpoints.store import save_context
from mobilerun_agents.agent.memory import read_memory
from mobilerun_agents.agent.memory.memory import build_planner_memory
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import (
    ActionRecord,
    AgentState,
    JevAbstention,
    Outcome,
    PlanningState,
    SubgoalResult,
)
from mobilerun_agents.config.user_config import MobileConfig


class AgentProbe(Workflow):
    @step
    async def start(self, ctx: Context[AgentState], ev: StartEvent) -> StopEvent:
        return StopEvent()


class PlanningProbe(Workflow):
    @step
    async def start(self, ctx: Context[PlanningState], ev: StartEvent) -> StopEvent:
        return StopEvent()


def test_fingerprint_is_stable_through_ui_checkpoint_round_trip():
    ui = UIState(
        [
            {
                "details": {"label": "shadowed", "class": "Button"},
                "label": "Confirm",
                "children": [None, {"details": "ignored", "value": "child", "children": "ignored"}],
            }
        ],
        "",
        "draft",
        {"packageName": "com.example", "activityName": "Main", "focusedElement": {"index": 2}},
        100,
        200,
    )
    expected = (
        "8659015c8e4999eb57975843c11575efcff12feb538625407b77412959835a29",
        "36617cb88907e8929f72949f5a223db216160b012282e2204e0dcbdcb3bd15ea",
    )
    vars(ui)["_has_single_focused_app_node"] = True
    snapshot = StepSnapshot(ui=ui)
    assert snapshot.fingerprint == expected
    payload = snapshot.model_dump_json()
    assert "_has_single_focused_app_node" not in json.loads(payload)["ui"]
    assert StepSnapshot.model_validate_json(payload).fingerprint == expected


def test_native_context_round_trip(tmp_path):
    async def check():
        pixels = b"fake screenshot bytes: checkpoint must never embed this"
        path = tmp_path / "screen.png"
        path.write_bytes(pixels)
        call = ToolSelection(
            tool_id="call_7",
            tool_name="tap",
            tool_kwargs={"index": 1, "nested": [None, {"value": True}]},
        )
        snapshot = StepSnapshot(
            ui=UIState([], "", "", {}, 100, 200), screenshot=ImageBlock(path=path)
        )
        expected = AgentState(
            goal="tap",
            current=snapshot,
            actions=[ActionRecord(step=1, call=call, ok=True, summary="tapped", before=snapshot)],
            outcome=Outcome(success=True, message=None, reason="complete", output={"n": 1}),
        )
        serializer = JsonSerializer()
        flow = AgentProbe()
        ctx = Context(flow)
        await ctx.store.set_state(expected)
        payload = ctx.to_dict(serializer=serializer)
        encoded = json.dumps(payload)
        assert str(path) in encoded
        assert pixels.decode() not in encoded and base64.b64encode(pixels).decode() not in encoded
        restored = await Context.from_dict(
            flow, json.loads(encoded), serializer=serializer
        ).store.get_state()
        assert restored.model_dump(mode="json") == expected.model_dump(mode="json")
        assert restored.outcome is not None and restored.outcome.message is None
        assert restored.actions[0].call == call
        assert restored.actions[0].before.ui.screen_width == 100
        assert (
            restored.current.screenshot.path == path and restored.current.screenshot.image is None
        )
        assert restored.current.fingerprint == snapshot.fingerprint

        await ctx.store.set(
            "current", StepSnapshot(ui=snapshot.ui, screenshot=ImageBlock(image=pixels))
        )
        transient = ctx.to_dict(serializer=serializer)
        assert base64.b64encode(pixels).decode() not in json.dumps(transient)
        restored_transient = Context.from_dict(flow, transient, serializer=serializer)
        assert await restored_transient.store.get("current.screenshot") is None

        plan = PlanningState(
            goal="tap",
            plan=["tap"],
            results=[
                SubgoalResult(
                    subgoal="tap",
                    success=True,
                    reason="complete",
                    before_screen=("shop", "List"),
                    after_screen=("shop", "Detail"),
                    screen_changed=True,
                    jev_abstention=JevAbstention(
                        choice="fixture-protected-value",
                        confidence=None,
                        target_confidence=None,
                        cutoff=0.5,
                        thought="fixture-protected-value",
                        choices="fixture-protected-value",
                    ),
                )
            ],
            planner_turns=1,
            executor_steps=4,
            actions=[
                expected.actions[0],
                ActionRecord(
                    step=4,
                    call=ToolSelection(tool_id="call_8", tool_name="wait", tool_kwargs={}),
                    ok=True,
                    summary="waited",
                    before=snapshot,
                ),
            ],
            current=snapshot,
            notes=["First child reached the target"],
            max_steps=12,
        )
        flow = PlanningProbe()
        ctx = Context(flow)
        await ctx.store.set_state(plan)
        saved_plan = await save_context(tmp_path / "context.json", ctx)
        plan_payload = json.dumps(saved_plan)
        assert pixels.decode() not in plan_payload
        assert base64.b64encode(pixels).decode() not in plan_payload
        restored_plan = Context.from_dict(flow, json.loads(plan_payload), serializer=serializer)
        restored_state = await restored_plan.store.get_state()
        assert restored_state.results[0].message is None
        assert restored_state.results[0].jev_abstention is not None
        abstention = restored_state.results[0].jev_abstention
        assert (
            abstention.choice
            == abstention.thought
            == abstention.choices
            == "fixture-protected-value"
        )
        assert [(record.step, record.call.tool_id) for record in restored_state.actions] == [
            (1, "call_7"),
            (4, "call_8"),
        ]

        async def render_memory(context, state):
            memory = build_planner_memory(MobileConfig(), state, "android")
            try:
                return (await read_memory(memory, context))[0].content
            finally:
                await memory.aclose()

        before = await render_memory(ctx, plan)
        after = await render_memory(restored_plan, restored_state)
        assert before == after
        assert 'Findings and notes: ["First child reached the target"]' in after
        assert "Latest executor subgoal result: Succeeded: tap" in after
        assert "Screen: shop/List → shop/Detail (changed)" in after

    asyncio.run(check())

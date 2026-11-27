import asyncio
import base64
import json

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.tools import ToolSelection
from mobilerun.core.devices import AndroidDevice
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from mobilerun.models.recording import Fingerprint
from mobilerun.utils.redaction import secret_scope
from workflows import Context, Workflow, step
from workflows.context.serializers import JsonSerializer
from workflows.events import StartEvent, StopEvent

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
from tests.unit.device_fixtures import MemoryDevice


def test_live_observation_keeps_prefilled_password_field_targetable():
    marker = "fixture-prefilled-password"

    class Phone(AndroidDevice):
        async def _get_ui_tree(self):
            return {
                "a11y_tree": {
                    "isPassword": True,
                    "resourceId": "login_password",
                    "className": "android.widget.EditText",
                    "text": marker,
                    "boundsInScreen": {"left": 10, "top": 10, "right": 70, "bottom": 30},
                    "children": [],
                },
                "phone_state": {"packageName": "example.app"},
                "device_context": {"screen_bounds": {"width": 100, "height": 100}},
            }

    snapshot = asyncio.run(StepSnapshot.observe(Phone("fake")))
    field = snapshot.ui.get_element(1)
    assert field is not None and field["text"] == "***"
    assert snapshot.ui.get_element_coords(1) == (40, 20)
    assert '1. EditText: "login_password", "***"' in snapshot.ui.formatted_text
    assert marker not in snapshot.ui.formatted_text and marker not in snapshot.ui.focused_text


def test_observation_keeps_image_but_masks_active_secret_in_live_text():
    marker = "fixture-active-scope-canary"

    class Screen(StateProvider):
        async def get_state(self):
            return UIState([{"text": marker}], marker, marker, {}, 100, 100)

    class Phone(MemoryDevice):
        async def context(self):
            return (ContextItem(source="fixture", kind="status", data={"text": marker}),)

        async def screenshot(self, hide_overlay=True):
            return b"live image"

    async def run():
        device = Phone()
        device._set_state_provider(Screen(device))
        with secret_scope(marker):
            snapshot = await StepSnapshot.observe(device, screenshot=True)
        assert "***" in snapshot.ui.formatted_text and marker not in snapshot.ui.formatted_text
        assert snapshot.ui.focused_text == "***"
        assert snapshot.ui.elements == [{"text": "***"}]
        assert snapshot.device_context == []
        assert snapshot.screenshot.image == base64.b64encode(b"live image")

    asyncio.run(run())


class AgentProbe(Workflow):
    @step
    async def start(self, ctx: Context[AgentState], ev: StartEvent) -> StopEvent:
        return StopEvent()


class PlanningProbe(Workflow):
    @step
    async def start(self, ctx: Context[PlanningState], ev: StartEvent) -> StopEvent:
        return StopEvent()


def test_fingerprint_is_stable_through_ui_model_round_trip():
    ui = UIState(
        [
            {
                "details": {"label": "shadowed", "class": "Button"},
                "resourceId": "confirm",
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
    expected = Fingerprint(app="com.example", activity="Main", elements=("id:confirm",))
    vars(ui)["_has_single_focused_app_node"] = True
    snapshot = StepSnapshot(ui=ui)
    assert snapshot.fingerprint == expected
    payload = snapshot.model_dump_json()
    assert "_has_single_focused_app_node" not in json.loads(payload)["ui"]
    assert StepSnapshot.model_validate_json(payload).fingerprint == expected


def test_native_context_round_trip():
    async def check():
        pixels = b"fake screenshot bytes: context payload must never embed this"
        call = ToolSelection(
            tool_id="call_7",
            tool_name="tap",
            tool_kwargs={"index": 1, "nested": [None, {"value": True}]},
        )
        snapshot = StepSnapshot(
            ui=UIState([], "", "", {}, 100, 200), screenshot=ImageBlock(image=pixels)
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
        assert pixels.decode() not in encoded and base64.b64encode(pixels).decode() not in encoded
        restored = await Context.from_dict(
            flow, json.loads(encoded), serializer=serializer
        ).store.get_state()
        assert restored.model_dump(mode="json") == expected.model_dump(mode="json")
        assert restored.outcome is not None and restored.outcome.message is None
        assert restored.actions[0].call == call
        assert restored.actions[0].before.ui.screen_width == 100
        assert restored.current.screenshot is None
        assert restored.current.fingerprint == snapshot.fingerprint

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
        saved_plan = ctx.to_dict(serializer=serializer)
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
        assert "First child reached the target" in after
        assert "Latest executor subgoal result: Succeeded: tap" in after
        assert "Screen: shop/List → shop/Detail (changed)" in after

    asyncio.run(check())

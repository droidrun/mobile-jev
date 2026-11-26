import asyncio

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.player import MacroPlayer

from mobilerun_agents.agent.checkpoints.macro_export import macro_data
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord
from tests.unit.device_fixtures import MemoryDevice


def record(name, args, ok=True):
    return ActionRecord(
        step=1,
        call=ToolSelection(tool_id="1", tool_name=name, tool_kwargs=args),
        ok=ok,
        summary="output",
        before=StepSnapshot(ui=UIState([{"text": "Settings"}], "", "", {}, 100, 200)),
    )


def test_macro_uses_before_ui_and_keeps_only_successful_replayable_calls():
    macro = macro_data(
        "open Settings",
        [
            record("tap", {"x": 12, "y": 34}),
            record("tap", {"x": 0, "y": 0}, ok=False),
            record("note", {"text": "temporary"}),
            record("type_secret", {"name": "entry", "package": "app", "field": "password"}),
        ],
        Redactor(),
    )
    assert [a["action_type"] for a in macro["actions"]] == ["tap", "type_secret"]
    assert macro["actions"][0]["pre_state"]["nodes"][0]["text"] == "Settings"
    assert macro["actions"][1]["name"] == "entry"
    assert macro["total_actions"] == 2


def test_macro_does_not_publish_masked_arguments_or_expanded_credentials():
    redact = Redactor()
    redact.add("fixture-sensitive")
    macro = macro_data(
        "type fixture-sensitive",
        [
            record("input_text", {"text": "fixture-sensitive"}),
            record("input_text", {"text": "***"}),
            record("type_secret", {"name": "entry", "text": "fixture-not-a-reference"}),
        ],
        redact,
    )
    assert macro["description"] == "type ***"
    assert not macro["actions"]


def test_base_can_load_the_export(tmp_path):
    import json

    path = tmp_path / "macro.json"
    path.write_text(json.dumps(macro_data("wait", [record("wait", {"seconds": 0})], Redactor())))
    assert MacroPlayer().load_macro_from_file(path)["actions"][0]["duration"] == 0


class GestureDevice(MemoryDevice):
    def __init__(self):
        super().__init__()
        self.gestures = []

    async def swipe(self, x1, y1, x2, y2, duration_ms=1000):
        self.gestures.append((x1, y1, x2, y2, duration_ms))

    async def drag(self, x1, y1, x2, y2, duration=3.0):
        self.gestures.append((x1, y1, x2, y2, duration))


@pytest.mark.parametrize(
    ("name", "timing", "expected_duration"),
    [
        ("swipe", {"duration_ms": 650}, 650),
        ("swipe", {}, 1000),
        ("drag", {"duration": 1.5}, 1.5),
        ("drag", {}, 3.0),
    ],
)
def test_exported_gesture_replays_with_the_recorded_coordinates_and_duration(
    name, timing, expected_duration, monkeypatch
):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("mobilerun.macro.player.asyncio.sleep", no_sleep)
    args = {"x1": 12, "y1": 34, "x2": 56, "y2": 78, **timing}
    [action] = macro_data("gesture", [record(name, args)], Redactor())["actions"]
    device = GestureDevice()
    assert asyncio.run(MacroPlayer(device).replay_action(action))
    assert device.gestures == [(12, 34, 56, 78, expected_duration)]
    assert not action.keys() & {"x1", "x2", "y1", "y2"}


def test_input_text_keeps_all_steps_and_omits_unsupported_options():
    records = [
        record("input_text", {"text": "normal", "clear": True, "stealth": False, "wpm": 0}),
        record("input_text", {"text": "stealthy", "stealth": True}),
        record("input_text", {"text": "paced", "wpm": 50}),
    ]
    actions = macro_data("type", records, Redactor())["actions"]
    assert [{k: v for k, v in action.items() if k != "pre_state"} for action in actions] == [
        {"action_type": "input_text", "text": "normal", "clear": True},
        {"action_type": "input_text", "text": "stealthy"},
        {"action_type": "input_text", "text": "paced"},
    ]

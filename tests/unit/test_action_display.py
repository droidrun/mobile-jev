import asyncio
import logging
from types import SimpleNamespace

from mobilerun.clients.cloud.tasks import _progress
from mobilerun.core.state.ui_state import UIState

from mobilerun.agents.operators import agent as operator
from mobilerun.agents.tools.tool_registry import ToolRegistry


def test_agent_actions_show_device_step_element_and_safe_typing(caplog):
    registry = ToolRegistry()

    async def action(**kwargs):
        return "ok"

    registry.register("click", action, {}, "")
    registry.register("type_text", action, {}, "")
    ui = UIState(
        [
            {
                "index": 8,
                "contentDescription": "+",
                "className": "android.widget.Button",
                "resourceId": "com.example:id/increment",
            }
        ],
        "",
        "",
        {},
        1,
        1,
    )
    ctx = SimpleNamespace(
        driver=SimpleNamespace(identifier="emulator-5554"),
        ui=ui,
        shared_state=SimpleNamespace(step_number=1),
    )
    with caplog.at_level(logging.INFO, logger="mobilerun.actions"):
        asyncio.run(registry.execute("click", {"index": 8}, ctx))
        asyncio.run(registry.execute("click", {"index": 9}, ctx))
        asyncio.run(registry.execute("type_text", {"text": "sensitive"}, ctx))
    assert [r.message for r in caplog.records] == [
        "emulator-5554 · step 1  tap Button[increment] #8",
        "emulator-5554 · step 1  tap #9",
        "emulator-5554 · step 1  type <redacted>",
    ]


def test_cloud_actions_skip_empty_progress_and_show_device(caplog):
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def iter_lines(self):
            yield 'data: {"type":"progress"}'
            yield 'data: {"step":3,"action":{"name":"click","args":{"index":10}}}'

    client = SimpleNamespace(
        tasks=SimpleNamespace(
            with_streaming_response=SimpleNamespace(attach=lambda *a, **kw: Stream())
        )
    )
    with caplog.at_level(logging.INFO, logger="mobilerun.actions"):
        asyncio.run(_progress(client, "task", "86f8d29f12345678", billing="no_commerce"))
    assert [r.message for r in caplog.records] == ["86f8d29f      · task  · step 3  tap #10"]


def test_agent_hint_not_visible_at_info(caplog, monkeypatch):
    monkeypatch.setattr(operator, "_agent_hint_logged", False)
    with caplog.at_level(logging.INFO, logger="mobilerun"):
        operator._log_agent_hint()
    assert not caplog.records


def test_action_never_shows_text_or_description():
    from mobilerun.logs.actions import format_action

    for field in ("text", "contentDescription", "description"):
        for label in ("Your code ABCDEF", "secret passphrase", "+", "Settings"):
            ui = SimpleNamespace(
                get_element=lambda index, field=field, label=label: {
                    field: label,
                    "className": "android.widget.Button",
                    "resourceId": "com.example:id/increment",
                }
            )
            assert format_action("tap", {"index": 8}, ui) == "tap Button[increment] #8"


def test_action_target_uses_available_class_id_and_index():
    from mobilerun.logs.actions import format_action

    for element, expected in (
        ({"className": "android.widget.EditText"}, "tap EditText #3"),
        ({"resourceId": "com.example:id/increment"}, "tap [increment] #3"),
        ({"text": "secret"}, "tap #3"),
    ):
        ui = SimpleNamespace(get_element=lambda index, element=element: element)
        assert format_action("tap", {"index": 3}, ui) == expected
    assert format_action("tap", {"x": 1, "y": 2}) == "tap (1, 2)"

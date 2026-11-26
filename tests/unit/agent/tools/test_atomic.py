from typing import Literal

import pytest
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents import Run, tool
from mobilerun_agents.agent.logs.diagnostics import Detail, sink
from tests.unit.device_fixtures import MemoryDevice, tool_run


@pytest.mark.anyio
async def test_atomic_logs_and_drops_a_product_above_the_configured_cap():
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.config.user_config import DecisionConfig

    @tool
    async def flags(run: Run, a: bool, b: bool) -> None:
        """Set two flags."""

    run = tool_run()
    run.registry = build_registry(run, [flags], None)
    events = []
    token = sink.set(events.append)
    try:
        assert not any(
            o.name == "flags"
            for o in await run.registry.atomic(
                run, config=DecisionConfig(max_schema_combinations=3)
            )
        )
        assert (
            len(
                [
                    o
                    for o in await run.registry.atomic(
                        run, config=DecisionConfig(max_schema_combinations=4)
                    )
                    if o.name == "flags"
                ]
            )
            == 4
        )
    finally:
        sink.reset(token)
    assert any(event.text == "dropped flags: 4 combinations exceed limit 3" for event in events)


@pytest.mark.anyio
async def test_atomic_drops_free_values_and_reports_each_once_per_run():
    from mobilerun_agents.agent.tools.registry import build_registry

    @tool
    async def choose(run: Run, value: Literal["a", "b"]) -> None:
        """Choose a value."""

    @tool
    async def input_text(run: Run, text: str) -> None:
        """Enter some text."""

    @tool
    async def tap(run: Run, x: int, y: int) -> None:
        """Tap a point."""

    run = tool_run(ui=UIState([], "", "", {}, 100, 100))
    run.registry = build_registry(run, [choose, input_text, tap], None).extend((), {"wait"})
    events = []
    token = sink.set(events.append)
    try:
        options = await run.registry.atomic(run, vision="off")
        assert await run.registry.atomic(run, vision="off") == options
    finally:
        sink.reset(token)
    assert [(option.name, option.parameters) for option in options] == [
        ("choose", {"value": "a"}),
        ("choose", {"value": "b"}),
    ]
    assert [(event.section, event.text) for event in events if isinstance(event, Detail)] == [
        ("tools", "dropped input_text: free str 'text'"),
        ("tools", "dropped tap: coordinates"),
    ]


@pytest.mark.anyio
async def test_atomic_expands_pinned_elements_apps_swipe_and_supported_buttons():
    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.tools.registry import build_registry

    class Phone(MemoryDevice):
        @property
        def supported_buttons(self):
            return {"back", "home"}

        async def list_apps(self, include_system=False):
            return [{"package": "app.settings", "label": "Settings"}]

    class Bench(Extension):
        default_name = "bench"

        @action
        async def choose(self, color: Literal["red", "blue"]) -> None:
            """Choose a color on the bench."""

    phone = Phone(supported_operations=frozenset({"tap", "swipe", "press_button", "start_app"}))
    run = tool_run(
        driver=phone,
        ui=UIState([{"index": 7, "text": "Settings", "bounds": "0,0,20,20"}], "", "", {}, 100, 100),
    )
    run.state_provider.capabilities = {"element_index"}
    run.state_provider.requires_coordinate_tools = False
    run.state_provider.requires_active_contract_for_coords = False
    await Bench().bind(phone)
    run.registry = build_registry(run, [], None)
    options = await run.registry.atomic(run, vision="off")
    by_name = {
        name: [o.parameters for o in options if o.name == name]
        for name in {o.name for o in options}
    }
    assert by_name["tap_element"] == [{"index": 7}]
    assert by_name["start_app"] == [{"package": "app.settings"}]
    assert by_name["press_button"] == [{"button": "back"}, {"button": "home"}]
    assert by_name["bench__choose"] == [{"color": "blue"}, {"color": "red"}]
    assert len(by_name["swipe"]) == 4
    down = next(o for o in options if o.label == "swipe down")
    assert down.parameters == {"x1": 50, "y1": 79, "x2": 50, "y2": 20, "duration_ms": 500.0}
    assert by_name["wait"] == [{}]


@pytest.mark.anyio
async def test_jev_excludes_by_base_name_and_can_complete_with_no_options():
    from mobilerun_agents.agent.action_selector.decision_model.base import (
        Decision,
        DecisionSelector,
    )
    from tests.unit.agent.selector_fixtures import selection_run
    from tests.unit.device_fixtures import fake_tools

    class Model:
        max_options = None
        locates_points = False

        async def decide(self, request):
            assert request.options == ()
            return Decision(success=True, message="Done")

    run = selection_run("Finish", tools=fake_tools("complete", "wait"))
    choice = await DecisionSelector(Model(), "off", exclude={"wait"}).select(run)
    assert choice.actions[0].name == "complete"
    assert choice.actions[0].parameters == {"success": True, "message": "Done"}
    assert "wait" in run.registry.by_name

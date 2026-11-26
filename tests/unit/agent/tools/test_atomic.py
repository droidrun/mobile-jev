from enum import Enum
from typing import Literal

import pytest
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents import Run, tool
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.diagnostics import Detail, private_values, sink
from tests.unit.device_fixtures import MemoryDevice, tool_run


@pytest.mark.anyio
async def test_cloud_swipes_send_integer_duration_for_llm_and_jev_choices():
    from mobilerun.errors import CloudValidationError

    from mobilerun_agents.agent.tools.registry import build_registry

    durations = []

    class Phone(MemoryDevice):
        @property
        def _is_cloud(self):
            return True

        async def swipe(
            self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
        ) -> None:
            if type(duration_ms) is not int:
                raise CloudValidationError(status=422)
            durations.append(duration_ms)
            if duration_ms == 999:
                raise CloudValidationError("backend fixture must not be echoed", status=422)

    phone = Phone(supported_operations=frozenset({"swipe"}))
    run = tool_run(driver=phone, ui=UIState([], "", "", {}, 1080, 2400))
    run.state_provider.capabilities = set()
    run.state_provider.requires_active_contract_for_coords = False
    run.state_provider.requires_coordinate_tools = False
    run.registry = build_registry(run, [], None)
    swipe = run.registry.by_name["swipe"]
    llm = {"x1": 846, "y1": 2086, "x2": 846, "y2": 1154, "duration_ms": 700.0}
    assert (await swipe.acall(**llm)).raw_output.success
    scroll = next(a for a in await run.registry.atomic_tools() if a.name == "swipe")
    assert (await swipe.acall(**scroll.parameters)).raw_output.success
    for duration, detail in [(700.5, "must be an integer"), (5, "must be at least 10")]:
        failed = await swipe.acall(**{**llm, "duration_ms": duration})
        assert not failed.raw_output.success
        assert failed.raw_output.summary == f"CloudValidationError: duration_ms {detail}"
    backend = await swipe.acall(**{**llm, "duration_ms": 999})
    assert backend.raw_output.summary == "CloudValidationError"
    assert durations == [700, 500, 999]


@pytest.mark.anyio
async def test_atomic_logs_and_drops_a_product_above_the_configured_cap():
    from mobilerun_agents.agent.tools.registry import build_registry

    @tool
    async def flags(run: Run, a: bool, b: bool) -> None:
        """Set two flags."""

    run = tool_run()
    run.registry = build_registry(run, [flags], None)
    events = []
    token = sink.set(events.append)
    try:
        run.tool_config.max_schema_combinations = 3
        assert not any(o.name == "flags" for o in await run.registry.atomic_tools())
        run.tool_config.max_schema_combinations = 4
        assert len([o for o in await run.registry.atomic_tools() if o.name == "flags"]) == 4
        run.tool_config.max_schema_combinations = 3
        assert not any(o.name == "flags" for o in await run.registry.atomic_tools())
        run.tool_config.max_schema_combinations = 4
        assert len([o for o in await run.registry.atomic_tools() if o.name == "flags"]) == 4
    finally:
        sink.reset(token)
    assert [event.text for event in events if "flags" in event.text] == [
        "dropped flags: 4 combinations exceed limit 3",
        "flags → 4 combinations (schema: bool 'a', bool 'b')",
        "dropped flags: 4 combinations exceed limit 3",
        "flags → 4 combinations (schema: bool 'a', bool 'b')",
    ]


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
    run.registry = build_registry(run, [choose, input_text, tap], None)
    events = []
    token = sink.set(events.append)
    try:
        options = await run.registry.atomic_tools({"wait", "complete"})
        assert await run.registry.atomic_tools({"wait", "complete"}) == options
    finally:
        sink.reset(token)
    assert [(option.name, option.parameters) for option in options] == [
        ("choose", {"value": "a"}),
        ("choose", {"value": "b"}),
    ]
    assert [(event.section, event.text) for event in events if isinstance(event, Detail)] == [
        ("tools", "choose → a, b (schema: Literal)"),
        ("tools", "dropped input_text: free str 'text'"),
        ("tools", "dropped tap: coordinates"),
    ]


@pytest.mark.anyio
async def test_atomic_expands_pinned_elements_apps_swipe_and_supported_buttons():
    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.tools.registry import build_registry

    class Phone(MemoryDevice):
        buttons = {"back", "home"}

        @property
        def supported_buttons(self):
            return self.buttons

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
    events = []
    token = sink.set(events.append)
    try:
        options = await run.registry.atomic_tools()
        assert await run.registry.atomic_tools() == options
        initial_ui = run.ui
        run.ui = UIState([], "", "", {}, 100, 100)
        phone.buttons = {"back", "enter", "home"}
        await run.registry.atomic_tools()
        run.ui = initial_ui
        phone.buttons = {"back", "home"}
        await run.registry.atomic_tools()
    finally:
        sink.reset(token)
    assert [event.text for event in events] == [
        "start_app → 1 app (exception)",
        "swipe → 4 scroll directions (exception)",
        "tap_element → 1 element (exception)",
        "complete → done / report failure (exception)",
        "bench__choose → blue, red (schema: Literal)",
        "press_button → back, home (schema: Literal)",
        "passed through: wait",
        "dropped tap: coordinates",
        "tap_element → 0 elements (exception)",
        "press_button → back, enter, home (schema: Literal)",
        "tap_element → 1 element (exception)",
        "press_button → back, home (schema: Literal)",
    ]
    by_name = {
        name: [o.parameters for o in options if o.name == name]
        for name in {o.name for o in options}
    }
    assert by_name["tap_element"] == [{"index": 7}]
    assert by_name["start_app"] == [{"package": "app.settings"}]
    assert by_name["press_button"] == [{"button": "back"}, {"button": "home"}]
    assert by_name["bench__choose"] == [{"color": "blue"}, {"color": "red"}]
    assert len(by_name["swipe"]) == 4
    down = next(o for o in options if o.label.startswith("scroll down"))
    assert down.label == "scroll down (show content further down; finger moves up)"
    assert down.parameters["y1"] > down.parameters["y2"]
    assert down.parameters == {"x1": 50, "y1": 79, "x2": 50, "y2": 20, "duration_ms": 500.0}
    assert by_name["wait"] == [{}]


@pytest.mark.anyio
@pytest.mark.parametrize("debug", [False, True])
async def test_atomic_console_hides_normal_output_and_redacts_schema_values(debug, caplog):
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import fake_tools

    class Mode(str, Enum):
        QUIET = "quiet"
        PRIVATE = "hidden-fixture"

    @tool
    async def set_mode(run: Run, mode: Mode) -> None:
        """Set the mode."""

    @tool(sensitive_parameters=frozenset({"token"}))
    async def authenticate(run: Run, token: str) -> None:
        """Authenticate."""

    run = tool_run()
    run.registry = build_registry(run, [set_mode, authenticate], None)
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog()
    hidden = private_values.set({"hidden-fixture"} if debug else None)
    token = sink.set(console.on_event if debug else None)
    try:
        await run.registry.atomic_tools(include=fake_tools("delegate"))
        await run.registry.atomic_tools()
        await run.registry.atomic_tools(include=fake_tools("delegate"))
    finally:
        sink.reset(token)
        private_values.reset(hidden)
    assert [message.strip() for message in caplog.messages if "tools:" in message] == (
        [
            "tools: complete → done / report failure (exception)",
            "tools: set_mode → ***, quiet (schema: enum)",
            "tools: passed through: delegate, wait",
            "tools: dropped authenticate: free str 'token'",
        ]
        if debug
        else []
    )


@pytest.mark.anyio
async def test_atomic_delegate_says_when_to_hand_the_step_to_the_llm():
    from tests.unit.agent.selector_fixtures import selection_run
    from tests.unit.device_fixtures import fake_tools

    run = selection_run("Type a value", tools=fake_tools("delegate"))
    [decision] = await run.registry.atomic_tools()
    assert decision.name == "delegate" and not decision.parameters
    assert (
        decision.label
        == "delegate: hand this step to the LLM (typing text, or nothing listed fits)"
    )


@pytest.mark.anyio
async def test_jev_excludes_by_base_name_and_can_complete_with_no_device_options():
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
            assert [o.action.parameters for o in request.options] == [
                {"success": True},
                {"success": False},
            ]
            return Decision(option_index=0, message="Done")

    run = selection_run("Finish", tools=fake_tools("complete", "wait"))
    choice = await DecisionSelector(Model(), "off", exclude={"wait"}).select(run)
    assert choice.actions[0].name == "complete"
    assert choice.actions[0].parameters == {"success": True, "message": "Done"}
    assert "wait" in run.registry.by_name

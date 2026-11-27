from typing import Literal

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.tools.base import tool
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.user_config import ToolsConfig
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import MemoryDevice


@pytest.mark.anyio
async def test_atomic_expands_only_finite_nonsensitive_choices():
    @tool
    async def mode(value: Literal["on", "off"], quiet: bool) -> None:
        """Change mode."""

    @tool
    async def free(value: str) -> None:
        """Read a value."""

    @tool(sensitive_parameters=frozenset({"value"}))
    async def secret(value: Literal["a", "b"]) -> None:
        """Use a secret."""

    phone = Turn().device
    registry = build_registry(phone, [mode, free, secret], ToolsConfig(max_schema_combinations=4))
    options = await registry.atomic_tools(exclude={"wait"})
    assert {a.name for a in options} == {"mode", "complete"}
    assert [a.parameters for a in options if a.name == "mode"] == [
        {"quiet": False, "value": "off"},
        {"quiet": False, "value": "on"},
        {"quiet": True, "value": "off"},
        {"quiet": True, "value": "on"},
    ]
    assert [a.parameters for a in options if a.name == "complete"] == [
        {"success": True},
        {"success": False},
    ]
    limited = build_registry(phone, [mode], ToolsConfig(max_schema_combinations=3))
    assert "mode" not in {a.name for a in await limited.atomic_tools()}


@pytest.mark.anyio
async def test_cloud_swipe_atomic_choice_executes_with_integer_duration():
    class Phone(MemoryDevice):
        _is_cloud = True

        async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 500) -> None:
            self.received = (x1, y1, x2, y2, duration_ms)

    phone = Phone(cloud_id="cloud", supported_operations=frozenset({"swipe"}))
    ui = UIState([], "", "", {}, 100, 200)
    turn = Turn(device=phone, ui=ui)
    choices = await turn.registry.atomic_tools(ui=ui)
    swipe = next(a for a in choices if a.name == "swipe")
    selected = ToolSelection(tool_id="x", tool_name=swipe.name, tool_kwargs=swipe.parameters)
    canonical = turn.registry.resolve(selected)
    assert canonical.tool_id == "x" and canonical.tool_name == "swipe"
    assert canonical.tool_kwargs == {"x1": 50, "y1": 40, "x2": 50, "y2": 159, "duration_ms": 500}
    output = await turn.registry.execute(selected, turn.ctx)
    assert output.success
    assert type(phone.received[-1]) is int and phone.received[-1] == 500


@pytest.mark.anyio
async def test_atomic_apps_prioritize_named_goal():
    class Phone(MemoryDevice):
        async def list_apps(self, include_system=False):
            assert include_system
            return [
                {"package": "org.maps", "label": "Maps"},
                {"package_name": "org.notes", "label": "Notes"},
            ]

    phone = Phone(supported_operations=frozenset({"start_app"}))
    registry = Turn(device=phone).registry
    choices = await registry.atomic_tools(goal="Read a note in Notes")
    assert [a.parameters for a in choices if a.name == "start_app"] == [
        {"package": "org.notes"},
        {"package": "org.maps"},
    ]

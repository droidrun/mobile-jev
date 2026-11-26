from types import SimpleNamespace
from unittest.mock import AsyncMock

from mobilerun.core.state.ui_state import UIState

from mobilerun_agents import ToolRegistry
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import AgentConfig
from tests.unit.device_fixtures import tool_run


def selection_run(
    goal, tools=(), screenshot=None, elements=None, parallel_tools=True, stream=False
):
    state = AgentState(goal=goal)
    run = tool_run(ui=UIState(elements or [], "screen", "", {}, 100, 200))
    run.state_provider.requires_coordinate_tools = screenshot is not None
    run.state_provider.requires_active_contract_for_coords = False
    run.config = AgentConfig(streaming=stream, parallel_tools=parallel_tools)
    run.registry = ToolRegistry(tuple(tools))
    run.device.screenshot = AsyncMock(return_value=screenshot)
    run.memory = SimpleNamespace(aget=AsyncMock(return_value=[]))
    run._state_snapshot = state
    return run


async def captured_turn(run):
    state = run.state
    return SimpleNamespace(
        goal=state.goal,
        history=list(state.actions),
        memory="\n".join(m.content or "" for m in await run.history()),
        by_name=run.registry.by_name,
        state=run.ui,
        registry=run.registry,
    )

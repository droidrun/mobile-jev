import json
from datetime import datetime, timezone

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.player import MacroPlayer
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.checkpoints import store
from mobilerun_agents.agent.checkpoints.macro_export import macro_data
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from tests.unit.device_fixtures import MemoryDevice


class Recorded(Workflow):
    @step
    async def record(self, ctx: Context[AgentState], ev: StartEvent) -> StopEvent:
        shot = StepSnapshot(
            at=datetime.now(timezone.utc),
            ui=UIState([], "", "", {}, 0, 0),
            device_context=[],
        )
        async with ctx.store.edit_state() as state:
            state.goal = "login with fixture-secret"
            state.actions = [
                ActionRecord(
                    step=1,
                    call=ToolSelection(
                        tool_id="1", tool_name="input_text", tool_kwargs={"text": "hello"}
                    ),
                    ok=True,
                    summary="typed hello; fixture-secret",
                    before=shot,
                )
            ]
        return StopEvent()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_native_checkpoint_restores_state_for_macro_replay(tmp_path):
    workflow = Recorded()
    handler = workflow.run()
    await handler
    path = tmp_path / "0001.json"
    await store.save_context(path, handler.ctx)
    checkpoint = path.read_text()
    assert (await handler.ctx.store.get_state()).goal == "login with fixture-secret"
    restored = Context[AgentState].from_dict(workflow, json.loads(checkpoint))
    state = await restored.store.get_state()
    assert state.goal == "login with fixture-secret"
    assert state.actions[0].summary == "typed hello; fixture-secret"
    macro = macro_data(state.goal, state.actions)
    assert macro["actions"][0]["pre_state"] is None
    device = MemoryDevice()
    assert await MacroPlayer(device).replay_macro(macro)
    assert device.inputs == ["hello"]

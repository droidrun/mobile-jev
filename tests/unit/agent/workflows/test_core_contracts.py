from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel

from mobilerun_agents.agent.action_selector.base import ActionSelection
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.agent.workflows.events import (
    Act,
    Done,
    Invalid,
    Observe,
    Select,
    StepFinished,
    UserMessage,
)


class Answer(BaseModel):
    count: int


def test_core_contracts():
    record = ActionRecord(
        step=1,
        name="click",
        args={"index": 3},
        ok=True,
        summary="ok",
        before=StepSnapshot(
            goal="count", step=1, ui=UIState([], "", "", {}, 0, 0), device_context=[]
        ),
    )
    state = AgentState(goal="count", actions=[record])
    assert (
        AgentState.model_validate_json(state.model_dump_json()).model_dump_json()
        == state.model_dump_json()
    )
    result = AgentResult[Answer](
        success=True, reason="done", answer="three", output=Answer(count=3), steps=1
    )
    assert AgentResult[Answer].model_validate_json(result.model_dump_json()) == result
    snapshot = StepSnapshot(
        goal="count",
        step=1,
        ui=UIState([], "", "", {}, 0, 0),
        device_context=[],
    )
    assert snapshot.fingerprint[1] == "" and Select() is not None
    assert Act(selection=ActionSelection()).selection.actions == []
    assert Done(result=result).result.answer == "three"
    assert Invalid(error="bad").error == "bad"
    assert StepFinished(action=record).action == record
    assert UserMessage(text="stop").text == "stop"
    assert Observe() is not None


def test_click_uses_observed_tree_not_macro_prestate():
    import asyncio
    from unittest.mock import AsyncMock, Mock, NonCallableMock

    from mobilerun.core.state.ui_state import UIState

    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import MemoryDevice, tool_run

    def tree(bounds):
        return UIState([{"index": 3, "bounds": bounds}], "", "", {}, 200, 200)

    class Device(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            assert (x, y) == (10, 20)

    async def run():
        newer = tree("70,80,90,100")
        recorder = Mock()
        device = Device(supported_operations=frozenset({"tap"}))
        ctx = tool_run(
            driver=device,
            ui=tree("0,10,20,30"),
            state_provider=NonCallableMock(
                capabilities={"element_index"}, get_state=AsyncMock(return_value=newer)
            ),
            macro_recorder=recorder,
        )
        result = (
            await build_registry(ctx, [], None).by_name["tap_element"].acall(index=3)
        ).raw_output
        assert result.success, result.summary
        recorder.record_action.assert_called_once_with(
            {"action_type": "tap", "x": 10, "y": 20}, pre_ui=newer
        )

    asyncio.run(run())


def test_operator_registry_is_filtered_and_describes_bound_actions():
    import asyncio
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.tools.base import complete, delegate
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.config.user_config import ToolsConfig
    from tests.unit.device_fixtures import MemoryDevice, tool_run

    class Extra(Extension):
        default_name = "extra"

        @action
        async def ping(self) -> str:
            """Read status."""
            return "ready"

    async def run():
        device = MemoryDevice(supported_operations=frozenset({"tap"}))
        provider = SimpleNamespace(
            capabilities={"element_index"}, use_normalized=False, requires_coordinate_tools=False
        )
        async with AsyncExitStack() as stack:
            run = tool_run(
                driver=device, state_provider=provider, config=ToolsConfig(exclude=["wait"])
            )
            ext = await Extra().bind(device)
            stack.push_async_callback(ext.unbind)
            merged = build_registry(run, [complete, delegate], Answer)
            registry = merged.by_name
            assert {
                "tap",
                "tap_element",
                "complete",
                "delegate",
                "extra__ping",
                "wait",
            } <= registry.keys()
            assert all(
                a.metadata.description.strip()
                and a.metadata.get_parameters_dict()["type"] == "object"
                for a in registry.values()
            )
            props = registry["complete"].metadata.get_parameters_dict()["properties"]
            assert set(props) == {"success", "message", "output"}
            assert props["output"]["properties"]["count"]["type"] == "integer"
            result = (await registry["extra__ping"].acall()).raw_output
            assert result.success and "ready" in result.summary
        assert not device.extensions

    asyncio.run(run())

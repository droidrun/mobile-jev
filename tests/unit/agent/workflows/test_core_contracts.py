import asyncio
from unittest.mock import AsyncMock, NonCallableMock

from llama_index.core.llms import ImageBlock
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import ToolsConfig


class Probe(Workflow):
    @step
    async def start(self, ctx: Context[AgentState], ev: StartEvent) -> StopEvent:
        return StopEvent()


class Answer(BaseModel):
    count: int


def test_comparison_uses_structural_identity_not_text_or_pixels():
    def snapshot(resource_id, text, image):
        return StepSnapshot(
            ui=UIState([{"resourceId": resource_id, "text": text}], "", "", {}, 100, 100),
            screenshot=ImageBlock(image=image),
        )

    first = snapshot("settings", "Settings", b"A")
    second = snapshot("settings", "Einstellungen", b"B")
    tree = snapshot("inbox", "Settings", b"A")
    missing = StepSnapshot(ui=UIState([], "", "", {}, 100, 100))
    assert first.compare(first) is False
    assert first.compare(second) is False
    assert first.compare(missing) is None
    assert first.compare(tree) is True


def test_click_uses_context_observation_not_newer_live_tree():
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import MemoryDevice

    def tree(bounds):
        return UIState([{"index": 3, "bounds": bounds}], "", "", {}, 200, 200)

    class Device(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            assert (x, y) == (10, 20)

    async def run():
        newer = tree("70,80,90,100")
        device = Device(supported_operations=frozenset({"tap"}))
        provider = NonCallableMock(
            spec=StateProvider,
            capabilities={"element_index"},
            get_state=AsyncMock(return_value=newer),
        )
        device._set_state_provider(provider)
        ctx = Context(Probe())
        await ctx.store.set_state(AgentState(current=StepSnapshot(ui=tree("0,10,20,30"))))
        registry = build_registry(device, [], ToolsConfig())
        result = await registry.execute(
            ToolSelection(tool_id="tap-1", tool_name="tap_element", tool_kwargs={"index": 3}), ctx
        )
        assert result.success, result.summary
        provider.get_state.assert_not_awaited()

    asyncio.run(run())


def test_operator_registry_is_filtered_and_describes_bound_actions():
    from contextlib import AsyncExitStack
    from types import SimpleNamespace

    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.tools.base import complete
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import MemoryDevice

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
            device._set_state_provider(provider)
            ext = await Extra().bind(device)
            stack.push_async_callback(ext.unbind)
            merged = build_registry(device, [complete], ToolsConfig(exclude=["wait"]), Answer)
            registry = merged.by_name
            assert {
                "tap",
                "tap_element",
                "complete",
                "extra__ping",
                "wait",
            } <= registry.keys()
            assert "delegate" not in registry
            assert all(
                a.metadata.description.strip()
                and a.metadata.get_parameters_dict()["type"] == "object"
                for a in registry.values()
            )
            props = registry["complete"].metadata.get_parameters_dict()["properties"]
            assert set(props) == {"success", "message", "output"}
            assert props["output"]["properties"]["count"]["type"] == "integer"
            call = ToolSelection(tool_id="ping-1", tool_name="extra__ping", tool_kwargs={})
            result = await merged.execute(call, Context(Probe()))
            assert result.success and "ready" in result.summary
        assert not device.extensions

    asyncio.run(run())

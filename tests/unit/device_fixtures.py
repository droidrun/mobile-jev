"""Repository in-memory Device; never connects to a phone or service."""

from mobilerun.core.devices.base import Device


class MemoryDevice(Device):
    def __init__(
        self,
        identifier: str = "memory",
        cloud_id: str | None = None,
        platform: str = "android",
        supported_operations: frozenset[str] = frozenset(),
        **device_options,
    ) -> None:
        super().__init__(**device_options)
        self._identifier = identifier
        self._cloud_id = cloud_id
        self._platform = platform
        self._supported_operations = supported_operations
        self.inputs: list[str] = []

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def cloud_id(self) -> str | None:
        return self._cloud_id

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def supported_operations(self) -> set[str]:
        return set(self._supported_operations)

    async def connect(self) -> None:
        return None

    async def _ensure_connected(self) -> None:
        return None

    async def _input_text_raw(
        self, text: str, clear: bool = False, stealth: bool = False, wpm: int = 0
    ) -> bool:
        self.inputs.append(text)
        return True


def fake_tools(*names):
    """Real compiled Action metadata, without binding or contacting a device."""
    from dataclasses import replace

    from mobilerun.core.extensions import Action
    from mobilerun.core.extensions.actions import Declaration

    from mobilerun_agents.agent.tools.base import ActionMetadata, RegistryTool

    async def noop() -> None:
        return None

    async def complete(output: dict[str, object] | None = None) -> None:
        return None

    async def indexed(index: int) -> None:
        return None

    async def coordinates(x: int, y: int) -> None:
        return None

    entries = []
    for name in names:
        device = name in {"tap", "tap_element"}
        fn = {"complete": complete, "tap_element": indexed, "tap": coordinates}.get(name, noop)
        action = replace(
            Action.compile("device" if device else "run", name, fn, Declaration()), name=name
        )
        entries.append(
            RegistryTool(
                ActionMetadata(
                    name=name,
                    description=action.description,
                    fn_schema=action._arguments,
                    ui_indexed=name == "tap_element",
                ),
                fn,
                action,
            )
        )
    return tuple(entries)


def device_turn(driver, ui, state_provider):
    from tests.unit.agent.action_selector.fixtures import Turn

    driver._set_state_provider(state_provider)
    return Turn(device=driver, ui=ui)


async def execute_tool(turn, name, **arguments):
    from llama_index.core.tools import ToolSelection

    return await turn.registry.execute(
        ToolSelection(tool_id="test", tool_name=name, tool_kwargs=arguments), turn.ctx
    )

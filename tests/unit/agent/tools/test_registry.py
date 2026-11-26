import asyncio

import pytest
from mobilerun.errors.operators import ExtensionActionError

from mobilerun_agents import Run, tool
from tests.unit.device_fixtures import MemoryDevice, tool_run


@pytest.mark.anyio
async def test_sensitive_native_input_bypasses_an_outer_recorder():
    from mobilerun.core.devices.recording import RecordingDevice

    from mobilerun_agents.agent.tools.registry import build_registry

    phone = MemoryDevice(supported_operations=frozenset({"input_text"}))
    outer = RecordingDevice(phone)
    run = tool_run(driver=outer)
    result = await build_registry(run, [], None).by_name["input_text"].acall(text="fixture-private")
    assert result.raw_output.success and phone.inputs == ["fixture-private"]
    assert "fixture-private" not in repr(outer.recorded_actions)


@pytest.mark.anyio
async def test_native_backend_failure_never_exposes_raw_message():
    from mobilerun_agents.agent.tools.registry import build_registry

    class Phone(MemoryDevice):
        async def open_url(self, url: str) -> None:
            raise RuntimeError("private backend body")

    run = tool_run(driver=Phone(supported_operations=frozenset({"open_url"})))
    result = (
        await build_registry(run, [], None).by_name["open_url"].acall(url="https://example.invalid")
    )
    assert not result.raw_output.success
    assert "RuntimeError" in result.content and "private backend body" not in result.content


def test_registry_respects_native_capabilities_and_preserves_non_ui_index():
    from mobilerun_agents.agent.action_selector.base import vision_registry
    from mobilerun_agents.agent.tools.registry import build_registry

    @tool
    async def inventory(run: Run, index: int) -> int:
        """Read an inventory entry by index."""
        return index

    run = tool_run(driver=MemoryDevice(supported_operations=frozenset({"tap"})))
    run.state_provider.capabilities = set()
    run.registry = build_registry(run, [inventory], None)
    assert {"tap", "inventory", "wait", "complete"} <= run.registry.by_name.keys()
    assert not {"tap_element", "swipe", "drag", "start_app"} & run.registry.by_name.keys()
    run.state_provider.capabilities = {"element_index"}
    run.registry = build_registry(run, [inventory], None)
    assert "tap_element" in run.registry.by_name
    assert "tap_element" not in vision_registry(run, "only").by_name
    assert "inventory" in vision_registry(run, "only").by_name


def test_classified_public_device_methods_match_base():
    import inspect

    from mobilerun.core.devices.base import Device

    from mobilerun_agents.agent.tools.registry import DEVICE_TOOL_DENYLIST

    assert {
        name
        for name, value in vars(Device).items()
        if not name.startswith("_") and inspect.iscoroutinefunction(value)
    } == (
        DEVICE_TOOL_DENYLIST - {"bind_extension"}
        | {
            "drag",
            "get_language",
            "get_timezone",
            "input_text",
            "list_apps",
            "open_url",
            "press_button",
            "set_clipboard",
            "start_app",
            "stop_app",
            "swipe",
            "tap",
            "tap_element",
            "wait",
        }
    )


@tool(sensitive_parameters=frozenset({"label"}))
async def label_phone(run: Run, label: str) -> str:
    """Label this device."""
    return f"{run.device.platform}: {label}"


def test_run_tool_schema_routing_and_collision():
    from mobilerun_agents.agent.tools.registry import build_registry

    device = MemoryDevice()
    run = tool_run(driver=device)
    entry = label_phone.compile(run)
    registry = build_registry(run, [label_phone], None)
    member = registry.by_name["label_phone"]
    assert member.metadata.fn_schema.model_json_schema() == entry.input_schema
    assert registry.extend(()).tools == registry.tools
    assert set(entry.input_schema["properties"]) == {"label"}
    assert entry.sensitive_parameters == {"label"}
    assert asyncio.run(member.acall(label="bench")).raw_output == f"{device.platform}: bench"
    with pytest.raises(ExtensionActionError):
        asyncio.run(member.acall(run="other", label="bench"))
    with pytest.raises(ValueError, match="label_phone"):
        registry.extend([member])


def test_exclude_names_and_sources_without_changing_existing_routes():
    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents import Source
    from mobilerun_agents.agent.tools.registry import build_registry

    class Bench(Extension):
        default_name = "bench"

        @action
        async def ping(self) -> str:
            """Read bench status."""
            return "ready"

    device = MemoryDevice(supported_operations=frozenset({"input_text"}))
    ext = asyncio.run(Bench().bind(device))
    try:
        base = build_registry(tool_run(driver=device), [label_phone], None)
        assert base.by_name["bench__ping"].metadata.source == "extension:bench"
        assert base.by_name["input_text"].metadata.source == Source.DEVICE
        filtered = base.extend((), exclude={Source.DEVICE, "label_phone"})
        assert set(filtered.by_name) == {"bench__ping", "complete", "wait"}
        with pytest.raises(KeyError):
            filtered.by_name["label_phone"]
        assert "bench__ping" not in base.extend((), exclude={"extension:bench"}).by_name
        ping = asyncio.run(base.by_name["bench__ping"].acall()).raw_output
        assert ping.success and "ready" in ping.summary
        assert {"input_text", "label_phone"} <= base.by_name.keys()
    finally:
        asyncio.run(ext.unbind())


def test_registry_rejects_extension_and_run_name_collision_even_when_excluded():
    from mobilerun.core.extensions import Extension, action

    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.config.user_config import ToolsConfig

    class Bench(Extension):
        default_name = "bench"

        @action
        async def ping(self) -> str:
            """Read bench status."""
            return "ready"

    @tool
    async def bench__ping(run: Run) -> None:
        """Ping this run."""

    device = MemoryDevice()
    ext = asyncio.run(Bench().bind(device))
    try:
        run = tool_run(driver=device, config=ToolsConfig(exclude=["bench__ping"]))
        with pytest.raises(ValueError, match="Duplicate tool name: bench__ping"):
            build_registry(run, [bench__ping], None)
    finally:
        asyncio.run(ext.unbind())


def test_tool_declaration_requires_async_documented_run_parameter():
    async def undocumented(run: Run) -> None:
        pass

    async def wrong(value: int) -> None:
        """Wrong first argument."""

    def sync(run: Run) -> None:
        """Not async."""

    for fn in (undocumented, wrong, sync):
        with pytest.raises(TypeError):
            tool(fn)


@pytest.mark.anyio
async def test_native_registry_pins_tap_element_and_records_a_replayable_guard(tmp_path):
    import json

    from mobilerun.core.state.base import StateProvider
    from mobilerun.core.state.ui_state import UIState
    from mobilerun.macro.models import MACRO_SCHEMA_VERSION
    from mobilerun.macro.player import MacroPlayer
    from mobilerun.macro.recorder import MacroRecorder

    from mobilerun_agents.agent.tools.registry import build_registry

    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            self.hit = (x, y)

    class Provider(StateProvider):
        capabilities = {"element_index"}

        async def get_state(self):
            return self.observation

    def ui(bounds):
        return UIState(
            [{"index": 7, "text": "Settings", "bounds": bounds}], "screen", "", {}, 100, 100
        )

    phone = Phone(
        supported_operations=frozenset({"tap", "input_text", "press_button", "start_app"})
    )
    provider = Provider(phone)
    pinned = provider.observation = ui("0,0,20,20")
    recorder = MacroRecorder()
    run = tool_run(driver=phone, ui=pinned, state_provider=provider, macro_recorder=recorder)
    registry = build_registry(run, [], None)
    assert {
        "tap_element",
        "tap",
        "input_text",
        "press_button",
        "start_app",
        "wait",
        "complete",
    } <= registry.by_name.keys()
    assert (
        not {"connect", "install_app", "press_key_code", "get_clipboard"} & registry.by_name.keys()
    )
    assert registry.by_name["wait"].metadata.get_parameters_dict().get("properties", {}) == {}

    provider.observation = ui("70,70,90,90")
    assert await phone.state.refresh() is not pinned
    output = (await registry.by_name["tap_element"].acall(index=7)).raw_output
    assert output.success and phone.hit == (10, 10)
    assert "tap_element" in output.summary
    assert recorder.actions[0]["pre_state"]

    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps({"macro_schema_version": MACRO_SCHEMA_VERSION, "actions": recorder.actions})
    )
    player = MacroPlayer(phone, delay_between_actions=0)
    macro = player.load_macro_from_file(path)
    assert await player.replay_macro(macro)
    assert phone.hit == (10, 10)


@pytest.mark.anyio
async def test_native_registry_vision_trims_by_schema():
    from mobilerun.core.state.ui_state import UIState

    from mobilerun_agents.agent.action_selector.base import vision_registry
    from mobilerun_agents.agent.tools.registry import build_registry

    @tool
    async def status(run: Run) -> str:
        """Read status."""
        return "ready"

    run = tool_run(
        driver=MemoryDevice(supported_operations=frozenset({"tap", "drag", "swipe"})),
        ui=UIState([{"index": 1}], "", "", {}, 80, 100),
    )
    run.state_provider.capabilities = {"element_index"}
    run.registry = build_registry(run, [status], None)
    coordinates = {"tap", "swipe", "drag"}
    assert not coordinates & vision_registry(run, "off").by_name.keys()
    assert coordinates | {"tap_element", "status"} <= vision_registry(run, "on").by_name.keys()
    only = vision_registry(run, "only").by_name
    assert coordinates | {"status", "wait", "complete"} <= only.keys()
    assert "tap_element" not in only

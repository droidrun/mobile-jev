import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_ime_id,
)

from mobilerun.agents.config.user_config import (
    DeviceConfig,
    MobileConfig,
    TelemetryConfig,
)


@pytest.mark.parametrize("ime", ["a>/x", "$(id)"])
def test_keyboard_cleanup_quotes_ime_shell_argument(monkeypatch, ime):
    from async_adbutils import adb

    from mobilerun.agents.clients.portal import keyboard

    device = SimpleNamespace(shell=AsyncMock())
    monkeypatch.setattr(adb, "device", AsyncMock(return_value=device))
    monkeypatch.setattr(keyboard, "portal_ime_id", lambda _: ime)
    asyncio.run(keyboard.disable_portal_keyboard("emulator-5554"))
    device.shell.assert_awaited_once_with(f"ime disable '{ime}'")


def test_disable_portal_keyboard_raises_wrapped_error(monkeypatch):
    from async_adbutils import adb

    from mobilerun.agents.clients.portal.keyboard import disable_portal_keyboard
    from mobilerun.agents.errors import KeyboardCleanupFailed

    original = RuntimeError("adb unavailable")

    class Device:
        async def shell(self, command):
            raise original

    async def device(serial):
        return Device()

    monkeypatch.setattr(adb, "device", device)
    with pytest.raises(KeyboardCleanupFailed) as exc:
        asyncio.run(disable_portal_keyboard("emulator-5554"))
    assert exc.value.__cause__ is original


def test_runner_cleanup_prefers_resolved_serial_and_skips_non_android(monkeypatch):
    from mobilerun.agents import runner

    called = []

    async def disable(serial):
        called.append(serial)

    monkeypatch.setattr(runner, "disable_portal_keyboard", disable)
    config = MobileConfig(device=DeviceConfig(serial="configured"))
    asyncio.run(runner._cleanup_portal_keyboard(config, "resolved"))
    assert called == ["resolved"]

    config.device.platform = "ios"
    asyncio.run(runner._cleanup_portal_keyboard(config, "ignored"))
    assert called == ["resolved"]


def test_runner_cleanup_falls_back_to_configured_serial(monkeypatch):
    from mobilerun.agents import runner

    called = []

    async def disable(serial):
        called.append(serial)

    monkeypatch.setattr(runner, "disable_portal_keyboard", disable)
    config = MobileConfig(device=DeviceConfig(serial="configured"))
    asyncio.run(runner._cleanup_portal_keyboard(config))
    assert called == ["configured"]


def test_runner_cleanup_skips_http_and_no_serial(monkeypatch):
    from mobilerun.agents import runner

    async def unexpected(serial):
        raise AssertionError("keyboard cleanup should be skipped")

    monkeypatch.setattr(runner, "disable_portal_keyboard", unexpected)
    asyncio.run(runner._cleanup_portal_keyboard(MobileConfig(), None))
    config = MobileConfig(device=DeviceConfig(serial="http://portal.test"))
    asyncio.run(runner._cleanup_portal_keyboard(config))


DISABLE_IME = f"ime disable {portal_ime_id(PORTAL_PACKAGE_NAME)}"


class FakeAdbDevice:
    def __init__(self, serial):
        self.serial = serial
        self.commands = []

    async def shell(self, command):
        self.commands.append(command)
        return ""


def _agent_module(monkeypatch, online_serial):
    from async_adbutils import adb

    from mobilerun.agents.workflows.droid import droid_agent as agent_module

    async def list_devices():
        return [SimpleNamespace(serial=online_serial, state="device")]

    async def device(serial=None):
        return FakeAdbDevice(serial)

    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(agent_module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(adb, "list", list_devices)
    monkeypatch.setattr(adb, "device", device)
    return agent_module


class FakeContext:
    def __init__(self):
        self.store = self

    def write_event_to_stream(self, event):
        return None

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


class ConnectFailed(Exception):
    pass


@pytest.mark.parametrize(
    ("serial", "expected"),
    [("emulator-5558", "emulator-5558")],
)
def test_agent_records_android_serial(monkeypatch, serial, expected):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")
    constructed = []

    class FakeAndroidDriver:
        def __init__(self, serial, **kwargs):
            constructed.append(serial)

        async def connect(self):
            raise ConnectFailed

    monkeypatch.setattr(agent_module, "AndroidDevice", FakeAndroidDriver)

    config = MobileConfig(
        device=DeviceConfig(serial=serial, auto_setup=False),
        telemetry=TelemetryConfig(enabled=False),
    )
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())
    assert agent.android_serial is None

    with pytest.raises(ConnectFailed):
        asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert constructed == [expected]
    assert agent.android_serial == expected
    assert config.device.serial == serial


def test_external_agent_records_android_serial(monkeypatch):
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")
    used = []

    async def run_external(device, **kwargs):
        used.append(device.serial)
        return {"success": True, "reason": "done"}

    monkeypatch.setattr(
        agent_module,
        "load_agent",
        lambda name: {"config": {}, "run": run_external},
    )

    config = MobileConfig(
        device=DeviceConfig(serial="emulator-5556"),
        telemetry=TelemetryConfig(enabled=False),
    )
    config.agent.name = "external-agent"
    agent = agent_module.MobileAgent("Open settings", config=config)

    asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert used == ["emulator-5556"]
    assert agent.android_serial == "emulator-5556"
    assert config.device.serial == "emulator-5556"


class PortalSetupFailed(Exception):
    pass


def test_agent_records_android_serial_when_portal_setup_fails(monkeypatch):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")

    async def failing_setup(device, **kwargs):
        raise PortalSetupFailed

    def unexpected_driver(*args, **kwargs):
        raise AssertionError("driver must not be built")

    monkeypatch.setattr(agent_module, "ensure_portal_ready", failing_setup)
    monkeypatch.setattr(agent_module, "AndroidDevice", unexpected_driver)

    config = MobileConfig(
        device=DeviceConfig(serial="emulator-5556"),
        telemetry=TelemetryConfig(enabled=False),
    )
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())

    with pytest.raises(PortalSetupFailed):
        asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

    assert agent.android_serial == "emulator-5556"
    assert config.device.serial == "emulator-5556"


@pytest.mark.parametrize("external", [False, True])
def test_missing_android_serial_requires_explicit_device_without_adb(monkeypatch, external):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch, "emulator-5556")

    async def unexpected_adb(*args, **kwargs):
        raise AssertionError("ADB must not be called")

    monkeypatch.setattr(agent_module.adb, "list", unexpected_adb)
    monkeypatch.setattr(agent_module.adb, "device", unexpected_adb)
    monkeypatch.setattr(agent_module, "load_agent", lambda name: {"config": {}})
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))
    if external:
        config.agent.name = "external-agent"
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())

    with pytest.raises(ValueError, match="No device given"):
        asyncio.run(agent.start_handler(FakeContext(), StartEvent()))

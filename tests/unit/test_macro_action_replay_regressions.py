import asyncio
from types import SimpleNamespace

import pytest
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import Extension, action
from mobilerun.errors.base import MacroError
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.recorder import MacroRecorder

from mobilerun.agents.tools.tool_registry import ToolRegistry
from tests.unit.device_fixtures import MemoryDevice


class Phone(MemoryDevice):
    def __init__(self):
        super().__init__(
            supported_operations=frozenset({"drag", "open_url", "stop_app", "press_key_code"})
        )
        self.calls = []

    async def open_url(self, url: str) -> None:
        self.calls.append(("open_url", url))

    async def stop_app(self, package: str) -> str:
        self.calls.append(("stop_app", package))
        return "stopped"

    async def press_key_code(self, key_code: int) -> None:
        self.calls.append(("press_key_code", key_code))


class Vault(Extension):
    default_name = "vault"

    def __init__(self):
        super().__init__()
        self.calls = []
        self.secret = "actual-secret-value"

    @action()
    async def type_field(self, package: str, field: str, name: str) -> None:
        """Type a stored field without exposing its value."""
        self.calls.append((package, field, name))
        assert self.secret == "actual-secret-value"

    @action(sensitive_parameters=frozenset({"value"}))
    async def send(self, value: str) -> None:
        """Send a sensitive value."""
        self.calls.append(value)


def test_registry_hides_raw_drag():
    registry = ToolRegistry()
    registry.register_device_actions(Phone())
    assert "device__drag" not in registry.tools


def test_registry_recording_replays_device_actions():
    async def run():
        recorded = RecordingDevice(Phone())
        registry = ToolRegistry()
        registry.register_device_actions(recorded)
        recorder = MacroRecorder()
        context = SimpleNamespace(driver=recorded, macro_recorder=recorder, ui=None)
        for name, args in [
            ("open_url", {"url": "https://example.com"}),
            ("stop_app", {"package": "app"}),
            ("press_key_code", {"key_code": 42}),
        ]:
            assert (await registry.execute("device__" + name, args, context)).success
        target = Phone()
        player = MacroPlayer(target)
        for entry in recorded.recorded_actions:
            await player.replay_action(entry)
        assert len(recorded.recorded_actions) == 3
        assert [item["action_type"] for item in recorder.actions] == [
            "open_url",
            "stop_app",
            "press_key_code",
        ]
        target.calls.clear()
        for entry in recorder.actions:
            await player.replay_action(entry)
        assert target.calls == recorded._inner.calls

    asyncio.run(run())


def test_redacted_extension_arguments_are_never_replayed_as_values():
    async def run():
        device = RecordingDevice(Phone())
        vault = Vault()
        await vault.bind(device)
        registry = ToolRegistry()
        registry.register_device_actions(device)
        recorder = MacroRecorder()
        result = await registry.execute(
            "vault__send",
            {"value": "actual-secret-value"},
            SimpleNamespace(driver=device, macro_recorder=recorder, ui=None),
        )
        assert result.success
        assert "actual-secret-value" not in str(device.recorded_actions)
        assert "actual-secret-value" not in str(recorder.actions)
        with pytest.raises(MacroError, match="redacted"):
            await MacroPlayer(device).replay_action(recorder.actions[0])
        assert vault.calls == ["actual-secret-value"]

    asyncio.run(run())


def test_extension_replay_uses_references_and_missing_binding_is_clear():
    async def run():
        recorded = RecordingDevice(Phone())
        vault = Vault()
        await vault.bind(recorded)
        registry = ToolRegistry()
        registry.register_device_actions(recorded)
        args = {"package": "app", "field": "password", "name": "work"}
        recorder = MacroRecorder()
        assert (
            await registry.execute(
                "vault__type_field",
                args,
                SimpleNamespace(driver=recorded, macro_recorder=recorder, ui=None),
            )
        ).success
        entries = recorded.recorded_actions
        assert len(entries) == 1 and entries[0]["arguments"] == args
        assert "actual-secret-value" not in str(entries)
        assert "actual-secret-value" not in str(recorder.actions)
        target = Phone()
        with pytest.raises(MacroError, match="vault"):
            await MacroPlayer(target).replay_action(entries[0])
        target_vault = Vault()
        await target_vault.bind(target)
        await MacroPlayer(target).replay_action(entries[0])
        assert target_vault.calls == [("app", "password", "work")]

    asyncio.run(run())

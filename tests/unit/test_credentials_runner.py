"""CLI configuration attaches providers before agent/macro execution."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun.cli.devices.base import selection
from mobilerun.core.credentials import FileCredentials
from mobilerun.core.devices.base import Device as DeviceMixin

from mobilerun.agents.config.user_config import MobileConfig


def test_runner_attaches_file_credentials_to_injected_device(tmp_path, monkeypatch):
    from mobilerun.agents import runner

    path = tmp_path / "credentials.yaml"
    path.write_text("login: private-value\n")
    path.chmod(0o600)
    config = MobileConfig()
    config.device.serial = "fake-serial"
    config.credentials.enabled = True
    config.credentials.file_path = str(path)
    config.telemetry.enabled = False
    device = DeviceMixin()
    monkeypatch.setattr(runner.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(runner, "_cleanup_portal_keyboard", AsyncMock())

    class Handler:
        async def stream_events(self):
            for item in ():
                yield item

        def __await__(self):
            async def result():
                return SimpleNamespace(success=True)

            return result().__await__()

    class Agent:
        def __init__(self, **kwargs):
            self.android_serial = None
            assert kwargs["driver"] is device
            assert isinstance(device.extensions["vault"], FileCredentials)
            assert "credentials" not in kwargs

        def run(self):
            return Handler()

    monkeypatch.setattr(runner, "MobileAgent", Agent)
    result = asyncio.run(
        runner.run_agent("login", device=device, target="fake-serial", device_kind="android")
    )
    assert result.success
    assert asyncio.run(device.extensions["vault"].list()) == [
        {"package": None, "name": "login", "fields": ["password"]}
    ]


@pytest.mark.parametrize("configured_path", [True, False])
def test_macro_cli_attaches_config_file_before_replay(tmp_path, monkeypatch, configured_path):
    from mobilerun.macro.cli import macro

    module = importlib.import_module("mobilerun.macro.cli.replay")
    secrets = tmp_path / "credentials.yaml"
    secrets.write_text("login: private-value\n")
    secrets.chmod(0o600)
    path = tmp_path / "macro.json"
    path.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [{"action_type": "type_secret", "secret": "login"}],
            }
        )
    )
    config = SimpleNamespace(
        credentials={
            "enabled": True,
            **({"file_path": str(secrets)} if configured_path else {}),
        },
        device=SimpleNamespace(platform="android", use_tcp=False),
    )
    device = DeviceMixin()
    device.input_text = AsyncMock(return_value=True)
    device._input_text_raw = device.input_text
    device.state.refresh = AsyncMock(
        return_value=SimpleNamespace(
            phone_state={
                "focusedElement": {"className": "EditText"},
                "isEditable": True,
            }
        )
    )
    monkeypatch.setattr(module.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(
        module, "get_settings", lambda: SimpleNamespace(local_credentials_path=secrets)
    )
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=device))
    result = CliRunner().invoke(macro, ["replay", str(path), "fake-serial", "--delay", "0"])
    assert result.exit_code == 0, (result.output, result.exception)
    device.input_text.assert_awaited_once_with("private-value")
    assert "private-value" not in result.output


def test_runner_never_binds_local_credentials_to_cloud(monkeypatch):
    from mobilerun.core.devices.cloud import CloudDevice

    from mobilerun.agents import runner

    config = MobileConfig()
    config.credentials.enabled = True
    config.telemetry.enabled = False
    device = CloudDevice("existing", api_key="test")
    monkeypatch.setattr(runner.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(runner, "_cleanup_portal_keyboard", AsyncMock())

    def forbidden(*args, **kwargs):
        raise AssertionError("local credentials bound to cloud")

    monkeypatch.setattr(runner, "FileCredentials", forbidden)

    class Handler:
        async def stream_events(self):
            for item in ():
                yield item

        def __await__(self):
            async def result():
                return SimpleNamespace(success=True)

            return result().__await__()

    class Agent:
        def __init__(self, **kwargs):
            self.android_serial = None

        def run(self):
            return Handler()

    monkeypatch.setattr(runner, "MobileAgent", Agent)

    async def run():
        try:
            result = await runner.run_agent(
                "login", device=device, target="existing", device_kind="android"
            )
            assert result.success
        finally:
            await device.close()

    asyncio.run(run())


def test_explicit_credentials_with_loose_permissions_report_actionable_error(
    tmp_path, monkeypatch, caplog
):
    from mobilerun.agents import runner

    path = tmp_path / "credentials.yaml"
    path.write_text("login: private-value\n")
    path.chmod(0o644)
    config = MobileConfig()
    config.credentials.enabled = True
    config.credentials.file_path = str(path)
    config.telemetry.enabled = False
    monkeypatch.setattr(runner.ConfigLoader, "load", lambda _: config)
    monkeypatch.setattr(runner, "_cleanup_portal_keyboard", AsyncMock())
    result = asyncio.run(
        runner.run_agent("login", device=DeviceMixin(), target="fake", device_kind="android")
    )
    assert not result.success
    assert "chmod 600" in caplog.text
    assert "private-value" not in caplog.text

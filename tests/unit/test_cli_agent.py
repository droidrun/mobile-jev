from click.testing import CliRunner
from mobilerun.cli.devices.base import selection


def test_legacy_run_remote_options_reach_real_device_selection(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from mobilerun.cli.app import cli
    from mobilerun.cli.devices.base import selection

    run_module = importlib.import_module("mobilerun.agents.cli.agent.run")
    devices = []

    class FakeRemote:
        def __init__(self, url, device_id):
            self.url, self.device_id = url, device_id
            devices.append(self)

        async def connect(self):
            pass

        async def close(self):
            pass

    async def fake_agent(**kwargs):
        assert kwargs["device"] is devices[0]
        assert kwargs["device_kind"] == "visual_remote"
        return SimpleNamespace(success=True, error=None, cloud_result=None)

    monkeypatch.setattr(selection, "RemoteDevice", FakeRemote)
    monkeypatch.setattr(run_module, "run_agent", fake_agent)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "goal",
            "--device",
            "http://remote.test",
            "--control-backend",
            "visual_remote",
            "--device-id",
            "phone-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (devices[0].url, devices[0].device_id) == ("http://remote.test", "phone-1")


def test_legacy_run_remote_uses_default_url_with_ios(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from mobilerun.cli.app import cli
    from mobilerun.cli.devices.base import selection
    from mobilerun_core_local.driver.visual_remote import VISUAL_REMOTE_DEFAULT_URL

    run_module = importlib.import_module("mobilerun.agents.cli.agent.run")
    devices = []

    class FakeRemote:
        def __init__(self, url, device_id):
            devices.append((url, device_id))

        async def connect(self):
            pass

        async def close(self):
            pass

    async def fake_agent(**kwargs):
        return SimpleNamespace(success=True, error=None, cloud_result=None)

    monkeypatch.setattr(selection, "RemoteDevice", FakeRemote)
    monkeypatch.setattr(run_module, "run_agent", fake_agent)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "goal",
            "--ios",
            "--control-backend",
            "visual_remote",
            "--device-id",
            "phone-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert devices == [(VISUAL_REMOTE_DEFAULT_URL, "phone-1")]


def test_agent_run_validates_saved_device(monkeypatch):
    import importlib
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import options

    from mobilerun.agents.cli.agent import agent

    run_module = importlib.import_module("mobilerun.agents.cli.agent.run")

    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(
        options.ConfigLoader,
        "load",
        lambda path: SimpleNamespace(
            device=SimpleNamespace(serial="saved", platform="android", use_tcp=None)
        ),
    )
    from mobilerun.cli.devices.base.discovery import DiscoveredDevice

    discovery = AsyncMock(return_value=[DiscoveredDevice("saved", "local", None, None, None)])
    monkeypatch.setattr(options, "discover_devices", discovery)
    run = AsyncMock(return_value=SimpleNamespace(success=True, error=None, cloud_result=None))
    monkeypatch.setattr(run_module, "run_agent", run)
    result = CliRunner().invoke(agent, ["run", "goal"])
    assert result.exit_code == 0, result.output
    assert run.await_args is not None
    assert run.await_args.kwargs["target"] == "saved"
    discovery.assert_awaited_once()


def test_agent_run_configures_cli_logging(monkeypatch):
    import importlib
    import logging
    from types import SimpleNamespace

    from mobilerun.logs.handlers import CLILogHandler

    from mobilerun.agents.cli.agent import agent

    run_module = importlib.import_module("mobilerun.agents.cli.agent.run")
    from unittest.mock import AsyncMock

    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=AsyncMock()))
    logger = logging.getLogger("mobilerun")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)

    async def fake_run(**kwargs):
        config = SimpleNamespace(
            logging=SimpleNamespace(rich_text=False, level="debug"),
            telemetry=SimpleNamespace(enabled=False),
        )
        kwargs["on_config"](config, True)
        assert any(isinstance(handler, CLILogHandler) for handler in logger.handlers)
        assert logger.level == logging.DEBUG
        logger.info("visible agent output")
        return SimpleNamespace(success=True, error=None, cloud_result=None)

    monkeypatch.setattr(run_module, "run_agent", fake_run)
    result = CliRunner().invoke(agent, ["run", "goal", "serial", "--android", "--debug"])
    assert result.exit_code == 0
    assert "visible agent output" in result.output


def test_agent_run_help_exposes_device_options():
    from mobilerun.agents.cli.agent import agent

    result = CliRunner().invoke(agent, ["run", "--help"])
    assert result.exit_code == 0
    for option in ("--vision-only", "--base_url", "--api_base", "--remote"):
        assert option in result.output
    for option in ("--android", "--ios", "--cloud", "--remote"):
        assert option in result.output
    assert "--device" in result.output


def test_run_forwards_options_and_exit_status(monkeypatch):
    import importlib

    from mobilerun.agents.cli.agent import agent

    run_module = importlib.import_module("mobilerun.agents.cli.agent.run")
    from unittest.mock import AsyncMock

    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=AsyncMock()))
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(success=False, error=None, cloud_result=None)

    monkeypatch.setattr(run_module, "run_agent", fake_run)
    result = CliRunner().invoke(
        agent,
        [
            "run",
            "open maps",
            "serial",
            "--android",
            "--vision-only",
            "--base_url",
            "http://example.test",
        ],
    )
    assert result.exit_code == 1
    assert calls[0]["command"] == "open maps"
    assert calls[0]["vision_only"] is True
    assert calls[0]["base_url"] == "http://example.test"
    assert calls[0]["target"] == "serial"
    assert calls[0]["device_kind"] == "android"


def test_agent_discovery_and_exclusive_device_flags(monkeypatch):
    from unittest.mock import AsyncMock

    from mobilerun.cli.devices.base import options

    from mobilerun.agents.cli.agent import agent

    monkeypatch.setattr(options, "discover_devices", AsyncMock(return_value=[]))
    runner = CliRunner()
    missing = runner.invoke(agent, ["run", "goal"])
    assert missing.exit_code == 1
    assert "No devices found." in missing.output
    multiple = runner.invoke(agent, ["run", "goal", "target", "--android", "--ios"])
    assert multiple.exit_code == 2
    assert "Choose at most one device type flag" in multiple.output


def test_runner_forces_vision_forwards_telemetry_and_cleans_keyboard(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from mobilerun.agents import runner as _runner

    def role():
        return SimpleNamespace(vision=False)

    config = SimpleNamespace(
        agent=SimpleNamespace(
            vision_only=False,
            manager=role(),
            executor=role(),
            fast_agent=role(),
            reasoning=False,
        ),
        device=SimpleNamespace(serial="serial", platform="android", control_backend=None),
        logging=SimpleNamespace(rich_text=False, debug=False, level="normal"),
        telemetry=SimpleNamespace(enabled=False),
        tracing=SimpleNamespace(enabled=False),
        credentials=SimpleNamespace(enabled=False),
    )
    configs = []
    cleaned = []
    forwarded = []
    monkeypatch.setattr(_runner.ConfigLoader, "load", lambda path: config)

    async def cleanup(serial):
        cleaned.append(serial)

    class Handler:
        async def stream_events(self):
            if False:
                yield None

        def __await__(self):
            async def result():
                return SimpleNamespace(success=True)

            return result().__await__()

    class Agent:
        android_serial = "resolved"

        def __init__(self, **kwargs):
            forwarded.append(kwargs["config"].device.serial)

        def run(self):
            return Handler()

    monkeypatch.setattr(_runner, "disable_portal_keyboard", cleanup)
    monkeypatch.setattr(_runner, "MobileAgent", Agent)
    assert (
        asyncio.run(
            _runner.run_agent(
                "test",
                target="explicit-serial",
                device_kind="android",
                vision_only=True,
                on_config=lambda config, debug: configs.append((config, debug)),
            )
        ).success
        is True
    )
    assert forwarded == ["explicit-serial"]
    assert all(
        role.vision
        for role in (
            config.agent.manager,
            config.agent.executor,
            config.agent.fast_agent,
        )
    )
    assert configs == [(config, False)]
    assert cleaned == ["resolved"]


def test_agent_run_sets_runtime_environment(monkeypatch):
    from mobilerun.agents import runner as _runner

    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("GRPC_ENABLE_FORK_SUPPORT", raising=False)

    class StopBeforeLoading(Exception):
        pass

    def load(_path):
        import os

        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["GRPC_ENABLE_FORK_SUPPORT"] == "false"
        raise StopBeforeLoading

    import asyncio

    import pytest

    monkeypatch.setattr(_runner.ConfigLoader, "load", load)
    with pytest.raises(StopBeforeLoading):
        asyncio.run(_runner.run_agent("test"))

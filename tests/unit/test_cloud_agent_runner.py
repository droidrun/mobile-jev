import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner
from mobilerun.cli.devices.base import selection
from mobilerun.errors import CloudAuthRequired, CloudTaskTimeout

from mobilerun.agents import runner
from mobilerun.agents.cli.agent import agent


@pytest.mark.parametrize(
    ("device", "flags"),
    [
        ("86f8d29f-9f59-4e42-ba02-1043c721774e", []),
        ("cloud-name", ["-c"]),
    ],
)
def test_cloud_cli_routes_to_cloud_do_and_warns_for_ignored_options(
    monkeypatch, tmp_path, device, flags
):
    calls = []
    config_file = tmp_path / "user.yaml"
    config_file.write_text("{}\n")

    class FakeCloudDevice:
        def __init__(self, *, device_id, base_url=None):
            calls.append(("construct", device_id, base_url))
            self._task_started_callback = None
            self.close = AsyncMock()

        async def do(self, goal, *, max_steps=None):
            calls.append(("do", goal, max_steps))
            self._task_started_callback("cloud-task-1")
            return SimpleNamespace(
                task_id="cloud-task-1",
                answer="done",
                success=True,
                steps=3,
                output={"x": 1},
            )

    importlib.import_module("mobilerun.agents.cli.agent.run")

    async def connect(target, **kwargs):
        return FakeCloudDevice(device_id=target, base_url=kwargs.get("base_url"))

    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(
        agent,
        [
            "run",
            "finish task",
            device,
            *flags,
            "--yes",
            "--steps",
            "8",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--vision",
            "--config",
            str(config_file),
        ],
    )
    assert result.exit_code == 0
    assert ("do", "finish task", 8) in calls
    assert ("construct", device, None) in calls
    displayed = result.output.splitlines()
    assert any("task_id=cloud-task-1" in line for line in displayed)
    assert any("Answer: done" in line for line in displayed)
    assert any("Success: True" in line for line in displayed)
    assert any("Steps: 3" in line for line in displayed)
    warning = next(line for line in displayed if "ignore local-only options" in line)
    assert all(flag in warning for flag in ("--provider", "--model", "--vision", "--config"))


@pytest.mark.parametrize(
    "error", [CloudAuthRequired("login required"), CloudTaskTimeout("timed out")]
)
def test_cloud_cli_exits_nonzero_for_auth_and_timeout(monkeypatch, error):
    class FakeCloudDevice:
        def __init__(self, **kwargs):
            raise error

    importlib.import_module("mobilerun.agents.cli.agent.run")

    async def connect(target, **kwargs):
        return FakeCloudDevice(device_id=target, base_url=kwargs.get("base_url"))

    monkeypatch.setattr(selection, "connect_target", connect)
    result = CliRunner().invoke(agent, ["run", "goal", "cloud-id", "-c", "--yes"])
    assert result.exit_code == 1
    if isinstance(error, CloudAuthRequired):
        assert (
            "Not logged in. Run `mobilerun account login` or set MOBILERUN_API_KEY."
            in result.output
        )
    else:
        assert str(error) in result.output


def test_cloud_runner_returns_result_without_terminal_output(monkeypatch, capsys):
    import asyncio

    class FakeCloudDevice:
        def __init__(self, **kwargs):
            self._task_started_callback = None

        async def do(self, goal, *, max_steps=None):
            if self._task_started_callback:
                self._task_started_callback("task-1")
            return SimpleNamespace(answer="done", success=True, steps=2, output=None)

    result = asyncio.run(runner.run_agent("goal", device=FakeCloudDevice(), device_kind="cloud"))
    assert result.success
    assert result.cloud_result.answer == "done"
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""

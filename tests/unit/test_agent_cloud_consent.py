import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner
from mobilerun.cli.base import menu, safety
from mobilerun.cli.devices.base import selection

from mobilerun_agents.cli.agent import agent


@pytest.fixture(autouse=True)
def configured_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-key")


@pytest.mark.parametrize(
    ("interactive", "answer", "yes", "allowed"),
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, True),
        (False, False, True, True),
        (True, False, True, True),
    ],
)
def test_cloud_agent_requires_consent_before_connect(
    monkeypatch, interactive, answer, yes, allowed
):
    module = importlib.import_module("mobilerun_agents.cli.agent.run")
    monkeypatch.setattr(menu, "is_interactive", lambda: interactive)
    prompt = Mock(return_value=answer)
    monkeypatch.setattr(safety, "confirm_prompt", prompt)
    connect = AsyncMock(return_value=AsyncMock())
    run = AsyncMock(return_value=SimpleNamespace(success=True, error=None, answer="ok"))
    monkeypatch.setattr(selection, "connect_target", connect)
    monkeypatch.setattr(module.AgentOperator, "execute", run)
    args = ["run", "goal", "phone-id", "--cloud"] + (["--yes"] if yes else [])
    result = CliRunner().invoke(agent, args)
    assert connect.await_count == int(allowed)
    assert run.await_count == int(allowed)
    assert result.exit_code == (1 if not interactive and not yes else 0), result.output
    assert "Interactive prompt failed" not in result.output
    assert prompt.call_count == int(interactive and not yes)
    if interactive:
        assert "may incur charges." in result.output
    if not interactive and not yes:
        assert result.exit_code == 1
        assert "Refusing to run a cloud task without confirmation; pass --yes." in result.output


@pytest.mark.parametrize(
    ("interactive", "answer", "yes", "allowed"),
    [
        (True, True, False, True),
        (True, False, False, False),
        (False, False, False, False),
        (True, False, True, True),
    ],
)
def test_task_stop_requires_consent_before_client(monkeypatch, interactive, answer, yes, allowed):
    module = importlib.import_module("mobilerun_agents.cli.agent.task")
    monkeypatch.setattr(menu, "is_interactive", lambda: interactive)
    prompt = AsyncMock(return_value=answer)
    monkeypatch.setattr(safety, "confirm_prompt_async", prompt)
    opened = []

    @asynccontextmanager
    async def client(task_id):
        opened.append(True)
        yield object()

    monkeypatch.setattr(module, "_client", client)
    status = AsyncMock(return_value=SimpleNamespace(status="running"))
    stop = AsyncMock(return_value=SimpleNamespace(cancelled=True))
    monkeypatch.setattr(module.tasks, "status", status)
    monkeypatch.setattr(module.tasks, "stop", stop)
    args = ["task", "stop", "abc"] + (["--yes"] if yes else [])
    result = CliRunner().invoke(agent, args)
    assert result.exit_code == (1 if not interactive and not yes else 0), result.output
    assert "Interactive prompt failed" not in result.output
    assert bool(opened) is allowed
    assert status.await_count == stop.await_count == int(allowed)
    assert prompt.await_count == int(interactive and not yes)
    if not interactive and not yes:
        assert "Refusing to stop the task without confirmation; pass --yes." in result.output


def test_local_agent_does_not_require_consent(monkeypatch):
    module = importlib.import_module("mobilerun_agents.cli.agent.run")
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=AsyncMock()))
    run = AsyncMock(return_value=SimpleNamespace(success=True, error=None, answer="ok"))
    monkeypatch.setattr(module.AgentOperator, "execute", run)
    result = CliRunner().invoke(agent, ["run", "goal", "serial", "--android"])
    assert result.exit_code == 0
    run.assert_awaited_once()


def test_short_yes_option_approves_cloud_run(monkeypatch):
    module = importlib.import_module("mobilerun_agents.cli.agent.run")
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    monkeypatch.setattr(selection, "connect_target", AsyncMock(return_value=AsyncMock()))
    run = AsyncMock(return_value=SimpleNamespace(success=True, error=None, answer="ok"))
    monkeypatch.setattr(module.AgentOperator, "execute", run)
    result = CliRunner().invoke(agent, ["run", "goal", "phone-id", "--cloud", "-y"])
    assert result.exit_code == 0
    run.assert_awaited_once()

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from mobilerun.clients.cloud import tasks
from mobilerun.models.billing import Billing
from mobilerun_sdk.types import (
    TaskGetStatusResponse,
    TaskGetTrajectoryResponse,
    TaskStopResponse,
)

from mobilerun_agents.cli.agent.group import agent


@pytest.fixture
def cloud(monkeypatch):
    from mobilerun_agents.cli.agent import task as module

    @asynccontextmanager
    async def client(*args, **kwargs):
        yield object()

    # cloud_client itself is a synchronous factory returning an async context manager.
    monkeypatch.setattr(module, "cloud_client", client)
    return module


def test_status_json_and_billing(cloud, monkeypatch):
    async def status(client, task_id, billing):
        assert task_id == "abc" and billing == Billing.NO_COMMERCE
        return TaskGetStatusResponse(
            status="completed",
            terminal=True,
            steps=3,
            message="Done",
            output={"ok": True},
        )

    monkeypatch.setattr(tasks, "status", status)
    result = CliRunner().invoke(agent, ["task", "status", "abc", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "completed" and data["steps"] == 3 and data["output"] == {"ok": True}
    assert data["terminal"] is True


def test_stop_requires_yes_and_handles_finished(cloud, monkeypatch):
    calls = []

    async def status(client, task_id, billing):
        return TaskGetStatusResponse(status="completed", terminal=True)

    async def stop(client, task_id, billing):
        calls.append(task_id)
        return TaskStopResponse(cancelled=True)

    monkeypatch.setattr(tasks, "status", status)
    monkeypatch.setattr(tasks, "stop", stop)
    runner = CliRunner()
    denied = runner.invoke(agent, ["task", "stop", "abc"])
    assert denied.exit_code != 0 and "--yes" in denied.output
    result = runner.invoke(agent, ["task", "stop", "abc", "--yes"])
    assert result.exit_code == 0 and "already" in result.output.lower()
    assert not calls


@pytest.mark.parametrize(
    "cancelled,exit_code,message",
    [(True, 0, "Stop requested"), (False, 1, "Error: Task abc was not stopped")],
)
def test_stop_running(cloud, monkeypatch, cancelled, exit_code, message):
    async def status(client, task_id, billing):
        return TaskGetStatusResponse(status="running", terminal=False)

    async def stop(client, task_id, billing):
        assert task_id == "abc" and billing == Billing.NO_COMMERCE
        return TaskStopResponse(cancelled=cancelled)

    monkeypatch.setattr(tasks, "status", status)
    monkeypatch.setattr(tasks, "stop", stop)
    result = CliRunner().invoke(agent, ["task", "stop", "abc", "--yes"])
    assert result.exit_code == exit_code, result.output
    assert message in result.output


def test_trajectory_json_and_download(cloud, monkeypatch, tmp_path):
    async def trajectory(client, task_id, billing):
        assert billing == Billing.NO_COMMERCE
        return TaskGetTrajectoryResponse.model_validate(
            {
                "trajectory": [
                    {
                        "event": "ScreenshotEvent",
                        "data": {"index": 4, "url": "https://example.test/shot"},
                    }
                ]
            }
        )

    monkeypatch.setattr(tasks, "trajectory", trajectory)

    class HTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url):
            assert url == "https://example.test/shot"
            return SimpleNamespace(content=b"image", raise_for_status=lambda: None)

    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kwargs: HTTP())
    result = CliRunner().invoke(
        agent, ["task", "trajectory", "abc", "--out", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["trajectory"][0]["event"] == "ScreenshotEvent"
    assert (tmp_path / "screenshot-4.png").read_bytes() == b"image"


def test_task_id_prompt(cloud, monkeypatch):
    from mobilerun.cli.base import menu

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "text_prompt", lambda label: "prompted-id")

    async def status(client, task_id, billing):
        assert task_id == "prompted-id"
        return TaskGetStatusResponse(status="running", terminal=False, steps=2)

    monkeypatch.setattr(tasks, "status", status)
    result = CliRunner().invoke(agent, ["task", "status"])
    assert result.exit_code == 0 and "running" in result.output


def test_not_logged_in(monkeypatch):
    from mobilerun.errors.cloud import CloudAuthRequired

    from mobilerun_agents.cli.agent import task as module

    def missing(*args, **kwargs):
        raise CloudAuthRequired(code="credentials_required")

    monkeypatch.setattr(module, "cloud_client", missing)
    result = CliRunner().invoke(agent, ["task", "status", "abc"])
    assert result.exit_code != 0 and "Not logged in" in result.output
    assert "mobilerun account login" in result.output


def test_not_found(cloud, monkeypatch):
    from mobilerun.errors.cloud import CloudResourceNotFound

    async def missing(client, task_id, billing):
        raise CloudResourceNotFound(operation="Poll cloud task", status=404)

    monkeypatch.setattr(tasks, "status", missing)
    result = CliRunner().invoke(agent, ["task", "status", "abc"])
    assert result.exit_code != 0 and "Task not found: abc" in result.output


def test_trajectory_without_out_does_not_download(cloud, monkeypatch):
    async def trajectory(client, task_id, billing):
        return TaskGetTrajectoryResponse.model_validate({"trajectory": []})

    monkeypatch.setattr(tasks, "trajectory", trajectory)

    def no_http(**kwargs):
        raise AssertionError("unexpected screenshot download")

    monkeypatch.setattr(cloud.httpx, "AsyncClient", no_http)
    result = CliRunner().invoke(agent, ["task", "trajectory", "abc"])
    assert result.exit_code == 0 and '"trajectory": []' in result.output

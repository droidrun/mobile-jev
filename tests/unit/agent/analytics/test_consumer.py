import asyncio
from unittest.mock import Mock
from uuid import UUID

import pytest
from mobilerun.config.user_config import TelemetryConfig

from mobilerun_agents.agent.analytics import consumer
from mobilerun_agents.agent.memory.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.result import AgentResult


def result():
    return AgentResult(success=True, reason="private", answer="secret", output=None, steps=4)


def test_visits_and_outcome_share_run_and_anonymous_identity(monkeypatch):
    client = Mock()
    monkeypatch.setattr(consumer, "_client", lambda: client)
    sink = consumer.RunAnalytics(TelemetryConfig(), "run")
    for step, activity in enumerate(["Home", "Home", "Settings", "Home"]):
        sink.on_observe(
            StepSnapshot(
                goal="secret",
                step=step,
                last_actions=[],
                package="app",
                activity=activity,
                ui_elements=[],
                device_context=[],
            )
        )
    asyncio.run(sink.finish(result()))
    asyncio.run(sink.finish(result()))
    calls = client.capture.call_args_list
    assert [c.args[0] for c in calls] == [
        "run_started",
        "app_visited",
        "app_visited",
        "app_visited",
        "run_finished",
    ]
    assert calls[-1].kwargs["properties"] == dict(
        run_id="run", success=True, steps=4, unique_packages_count=1, unique_activities_count=2
    )
    assert "secret" not in str(calls) and "private" not in str(calls)
    identity = calls[0].kwargs["distinct_id"]
    assert UUID(identity) and all(c.kwargs["distinct_id"] == identity for c in calls)
    assert all(c.kwargs["properties"]["run_id"] == "run" for c in calls)
    client.flush.assert_called_once()
    asyncio.run(consumer.RunAnalytics(TelemetryConfig(), "other").finish(result()))
    assert client.capture.call_args.kwargs["distinct_id"] == identity


@pytest.mark.parametrize("enabled,env", [(True, "no"), (False, "yes")])
def test_opt_out_never_creates_client_or_identity(enabled, env, monkeypatch):
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", env)
    monkeypatch.setattr(consumer, "_client", lambda: pytest.fail("client created"))
    monkeypatch.setattr(consumer, "_identity", lambda: pytest.fail("identity created"))
    sink = consumer.RunAnalytics(TelemetryConfig(enabled=enabled), "disabled")
    asyncio.run(sink.finish(result()))

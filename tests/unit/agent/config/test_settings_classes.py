"""Environment snapshots preserve the existing runtime configuration rules."""

import pytest
from mobilerun.config import settings

from mobilerun_agents.config import settings as agent_settings


def test_factories_return_fresh_snapshots(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", "first.yaml")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "first")
    old_settings = settings.get_settings()
    agent_old_settings = agent_settings.get_agent_settings()
    monkeypatch.setenv("MOBILERUN_CONFIG", "second.yaml")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "second")
    assert old_settings.MOBILERUN_CONFIG == "first.yaml"
    assert settings.get_settings().MOBILERUN_CONFIG == "second.yaml"
    assert agent_old_settings.PHOENIX_PROJECT_NAME == "first"
    assert agent_settings.get_agent_settings().PHOENIX_PROJECT_NAME == "second"


@pytest.mark.parametrize("value", ["", "true", "TRUE", "1", "yes", "y", "on", "maybe", " true "])
def test_telemetry_flag_accepts_strings(monkeypatch, value):
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", value)
    snapshot = agent_settings.get_agent_settings()
    assert snapshot.telemetry_enabled == (value.lower() in ("true", "1", "yes", "y"))


def test_nonflag_empty_values_and_defaults(monkeypatch):
    for name in (
        "MOBILERUN_CONFIG",
        "MOBILERUN_AUTH_URL",
        "MOBILERUN_CLOUD_BASE_URL",
        "PHOENIX_URL",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("LANGFUSE_HOST", "legacy-host")
    snapshot = settings.get_settings()
    agent_snapshot = agent_settings.get_agent_settings()
    assert snapshot.MOBILERUN_CONFIG == ""
    assert snapshot.MOBILERUN_AUTH_URL == "https://cloud.mobilerun.ai/api/auth"
    assert snapshot.MOBILERUN_CLOUD_BASE_URL == "https://api.mobilerun.ai/v1"
    assert agent_snapshot.PHOENIX_URL == ""
    assert agent_snapshot.langfuse_url == "legacy-host"
    monkeypatch.delenv("PHOENIX_URL")
    assert agent_settings.get_agent_settings().PHOENIX_URL == "http://0.0.0.0:6006"


def test_explicit_aliases_bypass_prefix_and_remain_case_sensitive(monkeypatch):
    monkeypatch.setenv("PHOENIX_URL", "exact-phoenix")
    monkeypatch.setenv("MOBILERUN_PHOENIX_URL", "prefixed-wrong")
    monkeypatch.setenv("MOBILERUN_MOBILERUN_CONFIG", "doubly-prefixed-wrong")
    monkeypatch.setenv("MOBILERUN_CONFIG", "exact.yaml")
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "14.5")
    snapshot = settings.get_settings()
    agent_snapshot = agent_settings.get_agent_settings()
    assert agent_snapshot.PHOENIX_URL == "exact-phoenix"
    assert snapshot.MOBILERUN_CONFIG == "exact.yaml"
    assert snapshot.MOBILERUN_CLOUD_API_TIMEOUT == 14.5

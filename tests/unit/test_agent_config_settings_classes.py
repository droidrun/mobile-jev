"""Environment snapshots preserve the existing runtime configuration rules."""

import pytest
from mobilerun.config import settings

from mobilerun.agents.config import env as agent_env
from mobilerun.agents.config import settings as agent_settings


def test_factories_return_fresh_snapshots(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "first-placeholder")
    monkeypatch.setenv("MOBILERUN_CONFIG", "first.yaml")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "first"))
    old_secret = agent_env.get_agent_secrets()
    old_settings = settings.get_settings()
    agent_old_settings = agent_settings.get_agent_settings()
    monkeypatch.setenv("OPENAI_API_KEY", "second-placeholder")
    monkeypatch.setenv("MOBILERUN_CONFIG", "second.yaml")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "second"))
    assert old_secret.OPENAI_API_KEY.get_secret_value() == "first-placeholder"
    assert agent_env.get_agent_secrets().OPENAI_API_KEY.get_secret_value() == "second-placeholder"
    assert old_settings.MOBILERUN_CONFIG == "first.yaml"
    assert settings.get_settings().MOBILERUN_CONFIG == "second.yaml"
    assert agent_old_settings.MOBILERUN_USER_CONFIG_DIR == tmp_path / "first"
    assert agent_settings.get_agent_settings().MOBILERUN_USER_CONFIG_DIR == tmp_path / "second"


@pytest.mark.parametrize("value", ["", "true", "TRUE", "1", "yes", "y", "on", "maybe", " true "])
def test_flags_have_distinct_truthy_sets_and_never_reject_strings(monkeypatch, value):
    for flag in ("TELEMETRY_ENABLED", "STREAM_SCREENSHOTS", "OAUTH_MANUAL"):
        monkeypatch.setenv(f"MOBILERUN_{flag}", value)
        monkeypatch.delenv(f"DROIDRUN_{flag}", raising=False)
    snapshot = agent_settings.get_agent_settings()
    assert snapshot.telemetry_enabled == (value.lower() in ("true", "1", "yes", "y"))
    assert snapshot.stream_screenshots == (value.lower() in ("true", "1"))
    assert snapshot.oauth_manual == (value.lower() in ("true", "1", "yes"))


def test_empty_and_absent_flag_fallbacks(monkeypatch):
    for flag in ("TELEMETRY_ENABLED", "STREAM_SCREENSHOTS", "OAUTH_MANUAL"):
        monkeypatch.setenv(f"MOBILERUN_{flag}", "")
        monkeypatch.setenv(f"DROIDRUN_{flag}", "true")
    snapshot = agent_settings.get_agent_settings()
    assert snapshot.telemetry_enabled is False
    assert snapshot.stream_screenshots is True
    assert snapshot.oauth_manual is True
    monkeypatch.delenv("MOBILERUN_TELEMETRY_ENABLED")
    monkeypatch.setenv("DROIDRUN_TELEMETRY_ENABLED", "")
    assert agent_settings.get_agent_settings().telemetry_enabled is True


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


def test_prefixed_tunables_and_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_CLOUD_API_TIMEOUT", "23.5")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT", "17")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL", "2")
    monkeypatch.setenv("MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP", "3")
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_APP_NAME", "not-the-app-name")
    snapshot = settings.get_settings()
    agent_snapshot = agent_settings.get_agent_settings()
    assert snapshot.MOBILERUN_CLOUD_API_TIMEOUT == 23.5
    assert snapshot.MOBILERUN_DEVICE_AUTH_REQUEST_TIMEOUT == 17
    assert snapshot.MOBILERUN_DEVICE_AUTH_MIN_POLL_INTERVAL == 2
    assert snapshot.MOBILERUN_DEVICE_AUTH_SLOW_DOWN_BUMP == 3
    assert agent_snapshot.oauth_credential_dir == tmp_path / "credentials"
    assert agent_snapshot.auth_profiles_path == tmp_path / "credentials" / "auth-profiles.json"
    assert (
        snapshot.mobilerun_cloud_credential_path
        == tmp_path / "credentials" / "mobilerun-cloud.json"
    )
    for provider in ("openai", "anthropic", "gemini", "grok"):
        assert (
            getattr(agent_snapshot, f"{provider}_oauth_credential_path")
            == agent_snapshot.auth_profiles_path
        )
    assert "app_name" not in settings.Settings.model_fields


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

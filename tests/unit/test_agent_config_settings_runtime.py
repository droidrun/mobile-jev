from mobilerun.config import settings

from mobilerun.agents.config import settings as agent_settings


def test_telemetry_getter_preserves_empty_primary(monkeypatch):
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "")
    monkeypatch.setenv("DROIDRUN_TELEMETRY_ENABLED", "yes")
    assert agent_settings.get_agent_settings().telemetry_enabled is False
    monkeypatch.delenv("MOBILERUN_TELEMETRY_ENABLED")
    assert agent_settings.get_agent_settings().telemetry_enabled is True


def test_runtime_setting_fallbacks(monkeypatch):
    monkeypatch.setenv("DROIDRUN_STREAM_SCREENSHOTS", "true")
    assert agent_settings.get_agent_settings().stream_screenshots is True
    monkeypatch.setenv("MOBILERUN_STREAM_SCREENSHOTS", "false")
    assert agent_settings.get_agent_settings().stream_screenshots is False
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "yes")
    assert agent_settings.get_agent_settings().oauth_manual is True


def test_settings_defaults_and_runtime_precedence(monkeypatch):
    from mobilerun.agents.config.user_config import AgentConfig

    monkeypatch.setenv("MOBILERUN_CONFIG", "new.yaml")
    assert settings.get_settings().MOBILERUN_CONFIG == "new.yaml"
    assert AgentConfig().max_steps == agent_settings.DEFAULT_MAX_STEPS == 15
    assert agent_settings.DEFAULT_MANAGER_SYSTEM_PROMPT == "assets/prompts/manager/system.jinja2"


def test_settings_flags_keep_distinct_empty_and_truthiness_rules(monkeypatch):
    monkeypatch.setenv("DROIDRUN_TELEMETRY_ENABLED", "yes")
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "")
    monkeypatch.setenv("DROIDRUN_STREAM_SCREENSHOTS", "true")
    monkeypatch.setenv("MOBILERUN_STREAM_SCREENSHOTS", "")
    monkeypatch.setenv("DROIDRUN_OAUTH_MANUAL", "yes")
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "")
    assert agent_settings.get_agent_settings().telemetry_enabled is False
    assert agent_settings.get_agent_settings().stream_screenshots is True
    assert agent_settings.get_agent_settings().oauth_manual is True
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "y")
    monkeypatch.setenv("MOBILERUN_STREAM_SCREENSHOTS", "yes")
    monkeypatch.setenv("MOBILERUN_OAUTH_MANUAL", "on")
    assert agent_settings.get_agent_settings().telemetry_enabled is True
    assert agent_settings.get_agent_settings().stream_screenshots is False
    assert agent_settings.get_agent_settings().oauth_manual is False
    monkeypatch.setenv("MOBILERUN_TELEMETRY_ENABLED", "maybe")
    assert agent_settings.get_agent_settings().telemetry_enabled is False


def test_empty_primary_config_falls_back_and_empty_phoenix_is_preserved(monkeypatch):
    monkeypatch.setenv("MOBILERUN_CONFIG", "")
    monkeypatch.setenv("PHOENIX_URL", "")
    assert settings.get_settings().MOBILERUN_CONFIG == ""
    assert agent_settings.get_agent_settings().PHOENIX_URL == ""


def test_langfuse_url_precedence_is_live(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "https://host.example.test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "")
    assert agent_settings.get_agent_settings().langfuse_url == "https://host.example.test"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://base.example.test")
    assert agent_settings.get_agent_settings().langfuse_url == "https://base.example.test"


def test_provider_base_url_override_is_read_at_runtime(monkeypatch):
    from mobilerun.agents.llm.providers.registry import resolve_provider_variant

    monkeypatch.setenv("MOBILERUN_DEEPSEEK_BASE_URL", "https://proxy.example.test/v1")
    from mobilerun.agents.llm import llm_picker

    assert (
        agent_settings.get_agent_settings().MOBILERUN_DEEPSEEK_BASE_URL
        == "https://proxy.example.test/v1"
    )
    assert (
        llm_picker.get_agent_settings().MOBILERUN_DEEPSEEK_BASE_URL
        == "https://proxy.example.test/v1"
    )
    assert resolve_provider_variant("ollama").base_url == "http://localhost:11434"


def test_posthog_host_override_is_read_at_runtime(monkeypatch):
    monkeypatch.setenv("MOBILERUN_POSTHOG_HOST", "https://posthog.example.test")
    assert (
        agent_settings.get_agent_settings().MOBILERUN_POSTHOG_HOST == "https://posthog.example.test"
    )


def test_legacy_config_environment_variable_is_ignored(monkeypatch):
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.setenv("DROIDRUN_CONFIG", "legacy.yaml")
    assert settings.get_settings().MOBILERUN_CONFIG is None

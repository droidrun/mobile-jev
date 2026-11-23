import pytest
from mobilerun.config.env import get_secrets, secret_value

from mobilerun.agents.config import env as agent_env


def test_secret_getter_reads_runtime_state(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert secret_value(agent_env.get_agent_secrets().OPENAI_API_KEY) is None
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    assert secret_value(agent_env.get_agent_secrets().OPENAI_API_KEY) == "test-placeholder"


def test_secret_names_include_dynamic_cloud_and_trace_keys(monkeypatch):
    monkeypatch.setenv("MOBILERUN_API_KEY", "placeholder")
    assert secret_value(get_secrets().MOBILERUN_API_KEY) == "placeholder"
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "placeholder")
    assert secret_value(agent_env.get_agent_secrets().LANGFUSE_SECRET_KEY) == "placeholder"


def test_settings_mask_secret_and_read_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "example-sensitive-placeholder")
    settings = agent_env.AgentSecrets()
    assert "example-sensitive-placeholder" not in repr(settings)
    assert "example-sensitive-placeholder" not in str(settings)
    assert (
        agent_env.get_agent_secrets().OPENAI_API_KEY.get_secret_value()
        == "example-sensitive-placeholder"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "changed-placeholder")
    assert agent_env.get_agent_secrets().OPENAI_API_KEY.get_secret_value() == "changed-placeholder"


def test_secret_settings_ignore_lowercase_and_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("openai_api_key", "lowercase-placeholder")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-placeholder\n")
    assert agent_env.AgentSecrets().OPENAI_API_KEY is None
    assert agent_env.get_agent_secrets().OPENAI_API_KEY is None


@pytest.mark.parametrize("name", agent_env.AgentSecrets.model_fields)
def test_agent_secret_aliases_are_masked(monkeypatch, name):
    alias = name
    monkeypatch.setenv(alias, "sensitive-placeholder")
    snapshot = agent_env.get_agent_secrets()
    assert getattr(snapshot, name).get_secret_value() == "sensitive-placeholder"
    assert "sensitive-placeholder" not in repr(snapshot)
    assert "sensitive-placeholder" not in str(snapshot)

import pytest
from click.testing import CliRunner

from mobilerun_agents.agent.auth import store
from mobilerun_agents.cli.agent import agent
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import DecisionProfile, DelegationConfig, LLMProfile


@pytest.fixture
def private_profile():
    config = ConfigLoader.load()
    config.agent.selector = "local"
    config.agent.delegation = DelegationConfig(primary="private", delegate="local")
    config.llm_profiles = {
        "private": LLMProfile(
            provider="openai_like",
            model="old",
            base_url="https://private.example/v1",
            kwargs={"context_window": 2048},
            temperature=0.6,
        ),
        "local": LLMProfile(provider="ollama", model="keep"),
    }
    ConfigLoader.save(config)
    return config


def test_same_provider_preserves_routing_and_other_profiles(private_profile, monkeypatch):
    monkeypatch.setenv("OPENAI_LIKE_API_KEY", "unit-key")
    result = CliRunner().invoke(agent, ["setup", "--provider", "openai_like", "--model", "new"])
    assert result.exit_code == 0, result.output
    private_profile.llm_profiles["private"].model = "new"
    assert ConfigLoader.load() == private_profile


def test_adding_provider_preserves_existing_endpoints(private_profile):
    args = ["setup", "--provider", "openai", "--model", "gpt-6-luna", "--api-key-stdin"]
    result = CliRunner().invoke(agent, args, input="unit-key\n")
    assert result.exit_code == 0, result.output
    private_profile.llm_profiles["openai"] = LLMProfile(provider="openai", model="gpt-6-luna")
    assert ConfigLoader.load() == private_profile


def test_set_default_preserves_distinct_delegate(private_profile):
    result = CliRunner().invoke(agent, ["setup", "--default", "ollama"])
    assert result.exit_code == 0, result.output
    assert ConfigLoader.load().agent.delegation.primary == "local"
    assert ConfigLoader.load().llm_profiles == private_profile.llm_profiles


def test_decision_selector_gains_an_active_llm_delegate():
    config = ConfigLoader.load()
    config.agent.selector = "jev"
    config.decision_profiles = {"jev": DecisionProfile()}
    config.llm_profiles = {}
    ConfigLoader.save(config)
    result = CliRunner().invoke(agent, ["setup", "--provider", "ollama", "--model", "local"])
    assert result.exit_code == 0, result.output
    config = ConfigLoader.load()
    assert config.agent.delegation is not None
    assert (config.agent.delegation.primary or config.agent.selector) == "jev"
    assert config.llm_profiles[config.agent.delegation.delegate] == LLMProfile(
        provider="ollama", model="local"
    )


def test_conflicting_delegate_fails_without_saving():
    config = ConfigLoader.load()
    config.agent.selector = "jev"
    config.agent.delegation = DelegationConfig(delegate="jev")
    config.decision_profiles["jev"] = DecisionProfile()
    ConfigLoader.save(config)
    result = CliRunner().invoke(
        agent,
        ["setup", "--provider", "openai", "--model", "gpt-6-luna", "--api-key-stdin"],
        input="unit-key\n",
    )
    assert (
        result.exit_code == 2 and "delegate" in result.output and "Configured" not in result.output
    )
    assert ConfigLoader.load() == config and not store.list_providers()


def test_identical_setup_reports_no_change(private_profile):
    private_profile.agent.delegation = None
    ConfigLoader.save(private_profile)
    result = CliRunner().invoke(agent, ["setup", "--provider", "ollama", "--model", "keep"])
    assert result.exit_code == 0 and "unchanged" in result.output.lower()
    assert ConfigLoader.load() == private_profile


@pytest.mark.parametrize("pinned", [False, True])
def test_jev_tracks_default_unless_explicitly_pinned(pinned):
    config = ConfigLoader.load()
    config.agent.selector = "jev"
    config.decision_profiles["jev"] = DecisionProfile()
    config.llm_profiles.clear()
    if pinned:
        config.llm_profiles["helper"] = LLMProfile(provider="anthropic", model="chosen")
        config.agent.delegation = DelegationConfig(delegate="helper")
    ConfigLoader.save(config)
    runner = CliRunner()
    assert (
        runner.invoke(
            agent, ["setup", "--provider", "ollama", "--model", "local", "--default"]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            agent,
            ["setup", "--provider", "openai", "--default", "--api-key-stdin"],
            input="fixture-key\n",
        ).exit_code
        == 0
    )
    config = ConfigLoader.load()
    assert config.default_llm == "openai" and config.agent.selector == "jev"
    assert config.agent.delegation.delegate == ("helper" if pinned else "openai")

from importlib import import_module

import pytest
import yaml
from click.testing import CliRunner
from mobilerun.cli.base import menu, prompts

from mobilerun_agents.agent.auth import store
from mobilerun_agents.cli.agent import agent
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import AgentConfig, MobileConfig


def invoke(*args, stdin=None):
    result = CliRunner().invoke(agent, args, input=stdin)
    assert "fixture-secret" not in result.output
    return result


def test_multiple_providers_and_either_default_syntax():
    assert invoke("setup", "--provider", "ollama").exit_code == 2
    assert invoke("setup", "--provider", "ollama", "--model", "local").exit_code == 0
    args = ["setup", "--provider", "openai", "--api-key-stdin"]
    assert invoke(*args, stdin="fixture-secret\n").exit_code == 0
    config = ConfigLoader.load()
    assert config.llm_profiles[config.agent.selector].provider == "ollama"
    assert {p.provider for p in config.llm_profiles.values()} == {"ollama", "openai"}
    assert invoke("setup", "--default", "openai").exit_code == 0
    assert ConfigLoader.load().agent.selector == "openai"
    assert invoke("setup", "--provider", "ollama", "--default").exit_code == 0
    assert ConfigLoader.load().agent.selector == "ollama"
    assert invoke("logout", "openai").exit_code == 0
    assert not store.list_providers()


@pytest.mark.parametrize("cancel", [False, True])
def test_menu_stages_keys_and_retries_empty_input(monkeypatch, cancel):
    ui = import_module("mobilerun_agents.cli.agent.wizard")
    answers = iter(["Add provider", "openai", "API key", "gpt-6-luna", None if cancel else "Done"])

    def pick(message, choices):
        label = next(answers)
        if message == "Authentication":
            assert [c.label for c in choices] == ["API key"]
        return next(c.value for c in choices if c.value == label) if label else None

    keys = iter(["", "fixture-secret"])
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(prompts, "select_prompt", pick)
    monkeypatch.setattr(ui, "read_secret", lambda: next(keys))

    def models(provider):
        assert provider == "openai"
        return ["gpt-6-luna"]

    monkeypatch.setattr(ui, "list_models", models)
    result = invoke("setup")
    assert (
        result.exit_code in (0, 130)
        and "empty" in result.output.lower()
        and "Usage:" not in result.output
    )
    if cancel:
        assert "Setup cancelled." in result.output and not store.list_providers()
        assert not ConfigLoader.get_user_config_path().exists()
    else:
        assert ConfigLoader.load().llm_profiles["openai"].model == "gpt-6-luna"
        assert store.load("openai").api_key == "fixture-secret"


def test_unknown_default():
    assert invoke("setup", "--default", "openai").exit_code == 2
    assert not ConfigLoader.get_user_config_path().exists()


def test_setup_saves_only_user_choices():
    assert invoke("setup", "--provider", "ollama", "--model", "local").exit_code == 0
    path = ConfigLoader.get_user_config_path()
    saved = yaml.safe_load(path.read_text())
    expected = {
        "agent": {"selector": "ollama"},
        "default_llm": "ollama",
        "llm_profiles": {"ollama": {"provider": "ollama", "model": "local"}},
    }
    assert saved == expected
    configured = MobileConfig.model_validate(expected)
    configured.agent.decision.max_response_repair_attempts = 7
    ConfigLoader.save(configured)
    assert ConfigLoader.load() == configured
    assert yaml.safe_load(path.read_text())["agent"] == {
        "selector": "ollama",
        "decision": {"max_response_repair_attempts": 7},
    }

    class LaterAgentConfig(AgentConfig):
        max_steps: int = 32

    class LaterConfig(MobileConfig):
        agent: LaterAgentConfig

    assert LaterConfig.from_yaml(path).agent.max_steps == 32


@pytest.mark.parametrize("mode", ["off", "on", "only"])
def test_explicit_vision_for_an_unknown_local_model(mode):
    assert (
        invoke("setup", "--provider", "ollama", "--model", "custom", "--vision", mode).exit_code
        == 0
    )
    assert ConfigLoader.load().llm_profiles["ollama"].vision == mode


def test_unknown_interactive_model_discards_staged_key(monkeypatch):
    ui = import_module("mobilerun_agents.cli.agent.wizard")
    choices = iter(["Add provider", "openai", "API key", "other…", "Done"])
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "pick", lambda *_: next(choices))
    monkeypatch.setattr(ui, "read_secret", lambda: "fixture-secret")
    monkeypatch.setattr(prompts, "text_prompt", lambda _: "unknown-version")
    result = invoke("setup")
    assert result.exit_code == 0 and "openai_like" in result.output
    assert not ConfigLoader.get_user_config_path().exists() and not store.list_providers()


@pytest.mark.parametrize(
    "provider,known", [("openai", "gpt-6-luna"), ("anthropic", "claude-sonnet-5")]
)
def test_setup_validates_native_model_before_saving(provider, known):
    args = ("setup", "--provider", provider, "--api-key-stdin", "--model")
    rejected = invoke(*args, "unknown-version", stdin="fixture-secret\n")
    assert rejected.exit_code == 2 and "openai_like" in rejected.output
    assert not ConfigLoader.get_user_config_path().exists() and not store.list_providers()
    assert invoke(*args, known, stdin="fixture-secret\n").exit_code == 0


def test_picker_reads_installed_model_lists_and_supports_free_text(monkeypatch):
    ui = import_module("mobilerun_agents.cli.agent.wizard")
    assert ui.list_models("openai").index("gpt-6-luna") < ui.list_models("openai").index("gpt-4o")
    assert "claude-sonnet-5" in ui.list_models("anthropic")
    assert ui.list_models("openai_like") == []
    answers = iter(["Add provider", "ollama", "None (local)", "other…", "Done"])
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "pick", lambda *_: next(answers))
    monkeypatch.setattr(prompts, "text_prompt", lambda _: "my-model")
    assert invoke("setup").exit_code == 0
    assert ConfigLoader.load().llm_profiles["ollama"].model == "my-model"


def test_provider_menu_actions_and_supported_methods(monkeypatch):
    ui = import_module("mobilerun_agents.cli.agent.wizard")
    choices = iter(
        [
            "Add provider",
            "anthropic",
            "Use environment variable ANTHROPIC_API_KEY",
            "claude-sonnet-5",
            "Add provider",
            "openrouter",
            "Sign in with browser",
            "A",
            "Set default",
            "openrouter",
            "Change model",
            "openrouter",
            "B",
            "Remove provider",
            "anthropic",
            "Done",
        ]
    )
    methods = []

    def select(title, rows):
        labels = [row.label for row in rows]
        if title == "Authentication":
            methods.append(labels)
        answer = next(choices)
        assert answer in [row.value for row in rows]
        return answer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(prompts, "select_prompt", select)
    monkeypatch.setattr(ui.oauth, "browser_key", lambda: "fixture-secret")
    monkeypatch.setattr(
        ui,
        "list_models",
        lambda provider: ["claude-sonnet-5"] if provider == "anthropic" else ["A", "B"],
    )
    result = invoke("setup")
    assert result.exit_code == 0 and "openrouter ✓ key saved default B" in " ".join(
        result.output.split()
    )
    assert methods == [
        ["Use environment variable ANTHROPIC_API_KEY", "API key"],
        ["API key", "Sign in with browser"],
    ]
    config = ConfigLoader.load()
    assert set(config.llm_profiles) == {config.agent.selector} == {"openrouter"}
    assert store.list_providers() == ["openrouter"]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, EOFError])
def test_cancel_at_key_prompt(monkeypatch, interrupt):
    ui = import_module("mobilerun_agents.cli.agent.wizard")
    answers = iter(["Add provider", "gemini", "API key"])

    def select(title, rows):
        if title == "Authentication":
            assert [r.label for r in rows] == ["API key"]
        return next(answers)

    def stop():
        raise interrupt()

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(prompts, "select_prompt", select)
    monkeypatch.setattr(ui, "read_secret", stop)
    result = invoke("setup")
    assert result.exit_code == 0 and result.output.endswith("Setup cancelled.\n")
    assert not ConfigLoader.get_user_config_path().exists() and not store.list_providers()

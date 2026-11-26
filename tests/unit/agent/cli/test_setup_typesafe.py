import pytest
import yaml
from click.testing import CliRunner
from mobilerun.cli.base import menu, prompts

from mobilerun_agents.agent.auth import store
from mobilerun_agents.cli.agent import agent
from mobilerun_agents.config.loader import ConfigLoader


@pytest.mark.parametrize("source", ["new", "saved", "environment"])
def test_interactive_typesafe_status_and_recommended_delegate(monkeypatch, source):
    from importlib import import_module

    wizard = import_module("mobilerun_agents.cli.agent.wizard")
    runner = CliRunner()
    assert (
        runner.invoke(agent, ["setup", "--provider", "ollama", "--model", "local"]).exit_code == 0
    )
    if source == "saved":
        store.save_key("typesafe", "fixture-secret")
    if source == "environment":
        monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-secret")
    status = {"new": "no key", "saved": "✓ key saved", "environment": "✓ TYPESAFE_API_KEY set"}[
        source
    ]
    transcript = []
    answers = iter(
        [
            "Add provider",
            "typesafe",
            {
                "new": "API key",
                "saved": "Use saved API key",
                "environment": "Use environment variable TYPESAFE_API_KEY",
            }[source],
            None,  # Accept the first (recommended) Jev mode.
            "Done",
        ]
    )

    def select(title, rows):
        transcript.append((title, [row.label for row in rows]))
        answer = next(answers)
        return (
            rows[0].value
            if answer is None
            else next(row.value for row in rows if row.value == answer)
        )

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(prompts, "select_prompt", select)
    monkeypatch.setattr(wizard, "read_secret", lambda: "fixture-secret")
    result = runner.invoke(agent, ["setup"])
    assert result.exit_code == 0, result.output
    assert any(
        "typesafe (Jev decision model)" in row and status in row
        for title, rows in transcript
        if title == "Provider"
        for row in rows
    )
    assert (
        "How should Jev run?",
        ["Jev + delegate to ollama/local (recommended)", "Jev only"],
    ) in transcript
    assert "fixture-secret" not in result.output + repr(transcript)
    config = ConfigLoader.load()
    assert config.agent.selector == "typesafe"
    assert config.agent.delegation.delegate == "ollama"
    assert config.default_llm == "ollama"
    assert (store.load("typesafe") is not None) == (source != "environment")


@pytest.mark.parametrize("choice", ["Jev only", "Jev + delegate to the default LLM (recommended)"])
def test_interactive_typesafe_without_an_llm(monkeypatch, choice):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-secret")
    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    answers = iter(
        ["Add provider", "typesafe", "Use environment variable TYPESAFE_API_KEY", choice, "Done"]
    )
    monkeypatch.setattr(prompts, "select_prompt", lambda *_: next(answers))
    result = CliRunner().invoke(agent, ["setup"])
    assert result.exit_code == 0, result.output
    if choice == "Jev only":
        assert ConfigLoader.load().agent.selector == "typesafe"
        assert ConfigLoader.load().agent.delegation is None
    else:
        assert "Add an LLM profile" in result.output
        assert not ConfigLoader.get_user_config_path().exists()
    assert "fixture-secret" not in result.output


def test_noninteractive_typesafe_modes_and_logout_fallback(monkeypatch):
    monkeypatch.setattr(menu, "is_interactive", lambda: False)
    runner = CliRunner()
    cmd = ["setup", "--provider", "typesafe", "--default", "--api-key-stdin"]
    missing = runner.invoke(agent, cmd, input="fixture-secret\n")
    assert missing.exit_code == 2 and "Add an LLM profile" in missing.output
    assert not ConfigLoader.get_user_config_path().exists() and not store.list_providers()
    only = runner.invoke(agent, [*cmd, "--jev-only"], input="fixture-secret\n")
    assert only.exit_code == 0, only.output
    assert ConfigLoader.load().agent.selector == "typesafe"
    assert ConfigLoader.load().agent.delegation is None
    assert (
        runner.invoke(agent, ["setup", "--provider", "ollama", "--model", "local"]).exit_code == 0
    )
    assert ConfigLoader.load().agent.delegation is None
    delegate = runner.invoke(agent, cmd, input="fixture-secret\n")
    assert delegate.exit_code == 0, delegate.output
    assert ConfigLoader.load().agent.delegation.delegate == "ollama"
    assert runner.invoke(agent, ["logout", "typesafe"]).exit_code == 0
    config = ConfigLoader.load()
    assert config.agent.selector == config.default_llm == "ollama"
    assert (
        config.agent.delegation is None
        and not config.decision_profiles
        and not store.list_providers()
    )
    assert "fixture-secret" not in missing.output + only.output + delegate.output


def test_setup_records_explicit_vision_mode(monkeypatch):
    result = CliRunner().invoke(
        agent, ["setup", "--provider", "ollama", "--model", "dummy", "--vision", "off"]
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(ConfigLoader.get_user_config_path().read_text())
    assert saved["llm_profiles"]["ollama"]["vision"] == "off"

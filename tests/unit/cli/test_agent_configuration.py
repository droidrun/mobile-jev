import pytest
import yaml
from click.testing import CliRunner
from mobilerun.cli.base import menu, prompts

from mobilerun_agents.agent.auth import store
from mobilerun_agents.cli.agent.group import agent
from mobilerun_agents.config.loader import ConfigLoader


def invoke(*args, input=None):
    return CliRunner().invoke(agent, args, input=input)


def ok(*args, input=None):
    result = invoke(*args, input=input)
    assert result.exit_code == 0, result.output
    return result.output


def add_llm(name):
    ok("profiles", "add", name, "--provider", "ollama", "--model", name)


def terminal(monkeypatch, steps):
    pending = iter(steps)

    def select(title, options):
        expected_title, answer = next(pending)
        assert title == expected_title
        if callable(answer):
            return answer(options)
        assert answer is None or answer in [option.value for option in options]
        return answer

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    monkeypatch.setattr(menu, "select_prompt", select)
    monkeypatch.setattr(prompts, "select_prompt", select)
    return pending


def test_provider_key_is_not_displayed_or_saved_in_yaml():
    key = "test-provider-secret"
    assert key not in ok("providers", "add", "typesafe", "--api-key-stdin", input=key + "\n")
    assert key not in ok("providers", "list")
    assert "typesafe" in ok("providers", "list")
    assert not ConfigLoader.get_user_config_path().exists()
    ok("providers", "remove", "typesafe")
    assert "typesafe" not in ok("providers", "list")


def test_openrouter_browser_key_is_saved_without_printing(monkeypatch):
    key = "fake-browser-key"
    module = "mobilerun_agents.cli.agent.providers"
    monkeypatch.setattr(f"{module}.menu.is_interactive", lambda: True)
    monkeypatch.setattr(f"{module}.oauth.browser_key", lambda: key)

    def pick_method(title, options):
        assert "Sign in with browser" in options
        return "Sign in with browser"

    monkeypatch.setattr(f"{module}.pick", pick_method)
    assert key not in ok("providers", "add", "openrouter")
    assert store.load("openrouter").api_key == key


def test_invalid_profile_replacement_preserves_file():
    add_llm("luna")
    path = ConfigLoader.get_user_config_path()
    original = path.read_bytes() + b"\n# keep this comment\n"
    path.write_bytes(original)
    result = invoke("profiles", "add", "luna", "--provider", "openai", "--model", "invalid")
    assert "does not recognize" in result.output
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "url",
    [
        "https://user:fixture-secret@example.invalid/v1",
        "https://@example.invalid/v1",
        "https://example.invalid/v1?API%5Fkey=fixture-secret",
        "https://example.invalid/v1?access_token=fixture-secret",
        "https://example.invalid/v1?X-Amz-Signature=fixture-secret",
        "https://example.invalid/v1?code=fixture-secret",
        "https://example.invalid/v1?api-version=2026-11-26",
        "https://example.invalid/v1?",
        "https://example.invalid/v1#fixture-secret",
        "https://example.invalid/v1#",
    ],
)
def test_profile_add_rejects_url_userinfo_query_or_fragment(url):
    result = invoke(
        "profiles",
        "add",
        "private",
        "--provider",
        "openai_like",
        "--model",
        "custom",
        "--base-url",
        url,
    )
    assert result.exit_code == 1 and "agent providers add" in result.output
    assert len(result.output.splitlines()) == 1
    assert "fixture-secret" not in result.output
    assert not ConfigLoader.get_user_config_path().exists()


@pytest.mark.parametrize(
    "field",
    [
        "kwargs: {api_key: fixture-secret}",
        "authorization: fixture-secret",
        "base_url: 'https://example.invalid/v1?password=fixture-secret'",
        "base_url: 'https://example.invalid/v1?code=fixture-secret'",
        "base_url: 'https://example.invalid/v1?api-version=2026-11-26'",
        "base_url: 'https://example.invalid/v1#fixture-secret'",
    ],
)
def test_configure_refuses_to_rewrite_profiles_with_credentials(field):
    path = ConfigLoader.get_user_config_path()
    path.parent.mkdir(parents=True)
    original = f"profiles: {{private: {{provider: openai_like, model: custom, {field}}}}}\n"
    path.write_text(original)
    result = invoke("configure", "quick", "--selector", "private")
    assert result.exit_code == 1 and "agent providers add" in result.output
    assert "fixture-secret" not in result.output and len(result.output.splitlines()) == 1
    assert path.read_text() == original


def test_unknown_yaml_provider_fails_during_config_load():
    path = ConfigLoader.get_user_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("profiles: {private: {provider: not-supported, model: custom}}\n")
    result = invoke("configure", "show")
    assert result.exit_code == 1 and "Unknown provider" in result.output
    assert "private" in result.output and "agent profiles add --help" in result.output


def test_profile_base_url_accepts_plain_endpoint():
    url = "https://example.invalid:8443/api/v1"
    ok(
        "profiles",
        "add",
        "private",
        "--provider",
        "openai_like",
        "--model",
        "custom",
        "--base-url",
        url,
    )
    assert ConfigLoader.load().profiles["private"].base_url == url


def test_profile_and_setup_crud_persists_only_choices(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    add_llm("luna")
    add_llm("sol")
    ok(
        "profiles",
        "add",
        "jev-on",
        "--provider",
        "typesafe",
        "--vision",
        "on",
        "--min-confidence",
        "0.7",
    )
    ok("configure", "fast", "--selector", "luna")
    ok("configure", "deep", "--planner", "sol", "--executor", "jev-on", "--delegate", "luna")
    ok("configure", "default", "fast")
    assert "jev-on" in ok("profiles", "list")
    assert "deep" in ok("configure", "show")
    assert "executor: jev-on" in ok("configure", "show", "deep")
    path = ConfigLoader.get_user_config_path()
    assert yaml.safe_load(path.read_text()) == {
        "profiles": {
            "luna": {"provider": "ollama", "model": "luna"},
            "sol": {"provider": "ollama", "model": "sol"},
            "jev-on": {"provider": "typesafe", "vision": "on", "min_confidence": 0.7},
        },
        "setups": {
            "fast": {"selector": "luna"},
            "deep": {"planner": "sol", "executor": "jev-on", "delegate": "luna"},
        },
        "default_setup": "fast",
    }
    before = path.read_bytes()
    assert "Unknown" in invoke("profiles", "remove", "luna").output
    assert path.read_bytes() == before
    ok("configure", "remove", "deep")
    ok("profiles", "remove", "jev-on")
    ok("configure", "remove", "fast")
    assert "default_setup" not in yaml.safe_load(path.read_text())


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--selector", "unknown"], "Unknown profile 'unknown'"),
        (["--planner", "jev", "--executor", "luna"], "planner must be an LLM profile"),
        (["--selector", "luna", "--delegate", "luna"], "requires a decision model profile"),
        (["--selector", "jev"], "No API key for typesafe"),
    ],
)
def test_configure_rejects_invalid_references_and_missing_credentials(flags, message):
    add_llm("luna")
    ok("profiles", "add", "jev", "--provider", "typesafe")
    path = ConfigLoader.get_user_config_path()
    before = path.read_bytes()
    result = invoke("configure", "example", *flags)
    assert result.exit_code != 0 and message in result.output
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "flags",
    [
        ["--provider", "typesafe", "--model", "jev"],
        ["--provider", "typesafe", "--reasoning", "high"],
        ["--provider", "ollama", "--model", "local", "--min-confidence", "0.3"],
        ["--provider", "openai", "--model", "not-a-known-model"],
    ],
)
def test_invalid_profile_is_not_written(flags):
    assert invoke("profiles", "add", "invalid", *flags).exit_code != 0
    assert not ConfigLoader.get_user_config_path().exists()


@pytest.mark.parametrize("choice", ["Cancel", "Done"])
def test_interactive_configure_saves_on_done_and_not_cancel(monkeypatch, choice):
    add_llm("luna")
    monkeypatch.setattr("mobilerun_agents.cli.agent.configure.menu.is_interactive", lambda: True)
    path = ConfigLoader.get_user_config_path()
    before = path.read_bytes()
    terminal(
        monkeypatch,
        [
            ("Setup structure", "Straight"),
            ("Selector profile", "luna"),
            ("Make default?", "No"),
            ("Save setup?", choice),
        ],
    )
    ok("configure", "new")
    if choice == "Cancel":
        assert path.read_bytes() == before
    else:
        assert ConfigLoader.load().setups["new"].selector == "luna"


def test_agent_menu_configure_uses_roles_not_jev_variants(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-key")
    add_llm("luna")
    ok(
        "profiles",
        "add",
        "jev",
        "--provider",
        "typesafe",
        "--vision",
        "on",
        "--min-confidence",
        "0.7",
    )

    def structure(options):
        assert [row.label for row in options] == ["Straight", "Manager + executor"]
        return "Manager + executor"

    def planner(options):
        assert [(row.value, row.hint) for row in options] == [("luna", "llm · ollama/luna")]
        return "luna"

    def executor(options):
        assert [(row.value, row.hint) for row in options] == [
            ("luna", "llm · ollama/luna"),
            ("jev", "decision · typesafe · vision on · min 0.7"),
        ]
        return "jev"

    def setup_name(title, default):
        assert (title, default) == ("Setup name", "default")
        return "deep"

    monkeypatch.setattr(prompts, "text_prompt", setup_name)
    terminal(
        monkeypatch,
        [
            ("agent: select a command", "configure"),
            ("Setup structure", structure),
            ("Planner profile", planner),
            ("Executor profile", executor),
            ("Delegate (optional)", "luna"),
            ("Make default?", "Yes"),
            ("Save setup?", "Done"),
        ],
    )
    ok()
    config = ConfigLoader.load()
    assert config.default_setup == "deep"
    assert config.setups["deep"].model_dump(exclude_none=True) == {
        "planner": "luna",
        "executor": "jev",
        "delegate": "luna",
    }


def test_configure_can_save_the_suggested_default_name(monkeypatch):
    add_llm("luna")
    monkeypatch.setattr(prompts, "text_prompt", lambda title, default: default)
    terminal(
        monkeypatch,
        [
            ("Setup structure", "Straight"),
            ("Selector profile", "luna"),
            ("Make default?", "Yes"),
            ("Save setup?", "Done"),
        ],
    )
    ok("configure")
    assert ConfigLoader.load().default_setup == "default"


def test_interactive_configure_can_clear_existing_default(monkeypatch):
    add_llm("luna")
    ok("configure", "fast", "--selector", "luna")
    ok("configure", "default", "fast")

    def no_default(options):
        assert "None (no default)" in [option.value for option in options]
        return "None (no default)"

    terminal(
        monkeypatch,
        [
            ("Setup structure", "Straight"),
            ("Selector profile", "luna"),
            ("Make default?", no_default),
            ("Save setup?", "Done"),
        ],
    )
    output = ok("configure", "new")
    assert "Default setup cleared" in output
    assert ConfigLoader.load().default_setup is None


def test_configure_default_command_and_default_setup_name_are_distinct():
    add_llm("luna")
    ok("configure", "default", "--selector=luna")
    ok("configure", "default", "default")
    assert ConfigLoader.load().default_setup == "default"


def test_configure_default_unset_removes_yaml_choice_and_reports_fallback():
    add_llm("luna")
    ok("configure", "fast", "--selector", "luna")
    ok("configure", "default", "fast")
    output = ok("configure", "default", "--unset")
    assert (
        output.strip()
        == "Default setup cleared; agent run uses a straight agent with the first LLM profile."
    )
    assert "default_setup" not in yaml.safe_load(ConfigLoader.get_user_config_path().read_text())
    assert "Default setup: (none)" in ok("configure", "show")


def test_configure_default_picker_can_clear_default(monkeypatch):
    add_llm("luna")
    ok("configure", "fast", "--selector", "luna")
    ok("configure", "default", "fast")

    def no_default(options):
        assert any(row.label == "None (no default)" for row in options)
        return ""

    terminal(monkeypatch, [("Default setup", no_default)])
    output = ok("configure", "default")
    assert "Default setup cleared" in output
    assert "default_setup" not in yaml.safe_load(ConfigLoader.get_user_config_path().read_text())


def test_configure_remove_current_default_reports_clearing():
    add_llm("luna")
    ok("configure", "fast", "--selector", "luna")
    ok("configure", "default", "fast")
    output = ok("configure", "remove", "fast")
    assert "Default setup cleared" in output
    assert "default_setup" not in yaml.safe_load(ConfigLoader.get_user_config_path().read_text())


def test_provider_submenu_prompts_for_provider_with_key_status(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-secret")

    def provider(options):
        row = next(row for row in options if row.value == "typesafe")
        assert row.hint == "TYPESAFE_API_KEY set"
        return "typesafe"

    terminal(
        monkeypatch,
        [
            ("agent: select a command", "providers"),
            ("agent providers: select a command", "add"),
            ("Provider", provider),
            ("Authentication", "Use existing key"),
        ],
    )
    assert "fixture-secret" not in ok()


@pytest.mark.parametrize("subgroup", ["providers", "profiles"])
def test_submenu_back_returns_to_agent_menu(monkeypatch, subgroup):
    def back(options):
        assert [row.label for row in options] == ["add", "list", "remove", "Back"]
        return options[-1].value

    terminal(
        monkeypatch,
        [
            ("agent: select a command", subgroup),
            (f"agent {subgroup}: select a command", back),
            ("agent: select a command", None),
        ],
    )
    ok()


def test_profile_other_model_uses_free_text(monkeypatch):
    values = {"Profile name": "local", "Model ID": "my-model"}
    monkeypatch.setattr(prompts, "text_prompt", lambda title: values[title])
    terminal(
        monkeypatch,
        [
            ("Provider", "ollama"),
            ("Model", "other…"),
            ("Vision", "Default"),
            ("Reasoning", "Default"),
        ],
    )
    ok("profiles", "add")
    assert ConfigLoader.load().profiles["local"].model == "my-model"


@pytest.mark.parametrize("provider", ["openai", "typesafe"])
def test_profile_submenu_prompts_for_model_or_decision_options(monkeypatch, provider):
    values = {"Profile name": "chosen", "Minimum confidence (optional)": "0.6"}
    monkeypatch.setattr(prompts, "text_prompt", lambda title: values[title])

    def model(options):
        assert "other…" in [row.value for row in options]
        assert "gpt-6-luna" in [row.value for row in options]
        return "gpt-6-luna"

    terminal(
        monkeypatch,
        [
            ("agent: select a command", "profiles"),
            ("agent profiles: select a command", "add"),
            ("Provider", provider),
            *(([("Model", model)] if provider == "openai" else [])),
            ("Vision", "off"),
            *(([("Reasoning", "low")] if provider == "openai" else [])),
        ],
    )
    ok()
    profile = ConfigLoader.load().profiles["chosen"]
    assert profile.vision == "off"
    if provider == "openai":
        assert profile.model == "gpt-6-luna" and profile.reasoning == "low"
    else:
        assert profile.min_confidence == 0.6


def test_configure_show_prompts_for_setup_in_terminal(monkeypatch):
    add_llm("luna")
    ok("configure", "first", "--selector", "luna")
    ok("configure", "second", "--selector", "luna")
    terminal(monkeypatch, [("Setup to show", "second")])
    output = ok("configure", "show")
    assert "second:" in output and "first:" not in output


def test_run_on_old_config_has_one_actionable_error():
    path = ConfigLoader.get_user_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("agent: {selector: old-secret}\n")
    result = invoke("run", "do something")
    assert result.exit_code == 1
    assert result.output.strip() == (
        "Error: Config uses old keys; run: mobilerun agent profiles add … / agent configure …"
    )

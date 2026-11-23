from click.testing import CliRunner
from mobilerun.cli.app import cli


def test_root_registers_llm_under_agent_only():
    runner = CliRunner()
    assert runner.invoke(cli, ["agent", "llm", "configure", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["agent", "llm", "login", "openai", "--help"]).exit_code == 0
    assert runner.invoke(cli, ["llm", "--help"]).exit_code == 2


def test_llm_help_lists_configure_and_login():
    from mobilerun.agents.cli.agent.llm import llm

    result = CliRunner().invoke(llm, ["--help"])
    assert result.exit_code == 0
    assert "configure" in result.output
    assert "login" in result.output


def test_provider_help_and_timeout_validation():
    from mobilerun.agents.cli.agent.llm import llm

    runner = CliRunner()
    for provider in ("openai", "anthropic", "gemini", "xai"):
        result = runner.invoke(llm, ["login", provider, "--help"])
        assert result.exit_code == 0, result.output
        for option in (
            "--credential-path",
            "--timeout",
            "--open-browser",
            "--no-browser",
        ):
            assert option in result.output
        for value in ("0", "-1", "nan", "inf"):
            result = runner.invoke(llm, ["login", provider, "--timeout", value])
            assert result.exit_code == 2
            assert "finite number greater than zero" in result.output


def test_provider_options_and_dispatch(monkeypatch):
    import importlib

    from mobilerun.agents.cli.agent.llm import llm

    calls = []
    runner = CliRunner()
    for provider, helper, args, expected in (
        (
            "openai",
            "_run_openai_oauth_login",
            [
                "--model",
                "m",
                "--callback-host",
                "localhost",
                "--callback-port",
                "123",
                "--callback-path",
                "/cb",
            ],
            {
                "model": "m",
                "callback_host": "localhost",
                "callback_port": 123,
                "callback_path": "/cb",
            },
        ),
        (
            "gemini",
            "_run_gemini_oauth_login",
            [
                "--model",
                "m",
                "--callback-host",
                "localhost",
                "--callback-port",
                "123",
                "--callback-path",
                "/cb",
            ],
            {
                "model": "m",
                "callback_host": "localhost",
                "callback_port": 123,
                "callback_path": "/cb",
            },
        ),
        (
            "xai",
            "_run_grok_oauth_login",
            ["--model", "m", "--device-code"],
            {"model": "m", "device_code": True},
        ),
        ("anthropic", "_run_anthropic_oauth_login", [], {}),
    ):
        module = importlib.import_module(f"mobilerun.agents.cli.agent.llm.login.{provider}")
        monkeypatch.setattr(module, helper, lambda **kwargs: calls.append(kwargs))
        result = runner.invoke(
            llm,
            [
                "login",
                provider,
                "--credential-path",
                "here",
                "--timeout",
                "17",
                "--no-browser",
                *args,
            ],
        )
        assert result.exit_code == 0, result.output
        assert calls[-1] == {
            "credential_path": "here",
            "timeout": 17.0,
            "open_browser": False,
            **expected,
        }


def test_anthropic_token_skips_oauth(monkeypatch):
    import importlib

    from mobilerun.agents.cli.agent.llm import llm

    module = importlib.import_module("mobilerun.agents.cli.agent.llm.login.anthropic")
    calls = []
    monkeypatch.setattr(
        module,
        "save_anthropic_setup_token",
        lambda path, token: calls.append((path, token)),
    )
    result = CliRunner().invoke(
        llm, ["login", "anthropic", "--credential-path", "here", "--token", "secret"]
    )
    assert result.exit_code == 0, result.output
    assert calls == [("here", "secret")]


def test_configure_forwards_options(monkeypatch):
    import importlib

    from mobilerun.agents.cli.agent.llm import llm

    module = importlib.import_module("mobilerun.agents.cli.agent.llm.configure")
    calls = []
    monkeypatch.setattr(
        module, "run_configure_wizard", lambda *args, **kwargs: calls.append(kwargs)
    )
    result = CliRunner().invoke(
        llm,
        [
            "configure",
            "--provider",
            "openai",
            "--auth-mode",
            "api_key",
            "--model",
            "m",
            "--api-key",
            "secret",
            "--base-url",
            "https://example.test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "provider": "openai",
            "auth_mode": "api_key",
            "model": "m",
            "api_key": "secret",
            "base_url": "https://example.test",
        }
    ]

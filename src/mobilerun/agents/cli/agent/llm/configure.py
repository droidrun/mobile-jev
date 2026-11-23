import click

from mobilerun.agents.cli.agent.llm.login.anthropic import _run_anthropic_oauth_login
from mobilerun.agents.cli.agent.llm.login.gemini import _run_gemini_oauth_login
from mobilerun.agents.cli.agent.llm.login.openai import _run_openai_oauth_login
from mobilerun.agents.cli.agent.llm.login.xai import _run_grok_oauth_login
from mobilerun.agents.cli.agent.llm.wizard import (
    ConfigureWizardCallbacks,
    run_configure_wizard,
)
from mobilerun.agents.cli.base.oauth_output import console


@click.command(name="configure")
@click.option(
    "--provider",
    type=str,
    default=None,
    help="Provider family (gemini, openai, anthropic, XAI, ollama, openai_like, minimax, zai).",
)
@click.option(
    "--auth-mode",
    type=str,
    default=None,
    help="Auth mode for the selected provider family.",
)
@click.option("--model", type=str, default=None, help="Model to configure.")
@click.option("--api-key", type=str, default=None, help="API key for API-key providers.")
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="Base URL override for compatible providers.",
)
def configure(
    provider: str | None,
    auth_mode: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Configure LLM provider, auth mode, and model.

    Use `mobilerun agent llm login <provider>` to log in via OAuth.
    """
    run_configure_wizard(
        console,
        ConfigureWizardCallbacks(
            run_openai_oauth_login=_run_openai_oauth_login,
            run_anthropic_oauth_login=_run_anthropic_oauth_login,
            run_gemini_oauth_login=_run_gemini_oauth_login,
            run_grok_oauth_login=_run_grok_oauth_login,
        ),
        provider=provider,
        auth_mode=auth_mode,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

from typing import Unpack

import click
from mobilerun.cli.base import render

from mobilerun.agents.cli.base.oauth_output import (
    BrowserLoginOptions,
    _print_oauth_login_success,
    _validate_oauth_timeout,
)
from mobilerun.agents.clients.oauth.anthropic import (
    run_anthropic_setup_token_oauth,
    save_anthropic_setup_token,
)
from mobilerun.agents.config.settings import get_agent_settings


def _run_anthropic_oauth_login(credential_path: str, **kwargs: Unpack[BrowserLoginOptions]) -> None:
    """Run the full Anthropic OAuth flow inline and save the token."""
    render.kv([("Action", "Starting Anthropic login")])
    run_anthropic_setup_token_oauth(
        credential_path=credential_path,
        on_output=render.message,
        **kwargs,
    )
    _print_oauth_login_success("Anthropic", credential_path)


@click.command("anthropic")
@click.option(
    "--credential-path",
    default=lambda: str(get_agent_settings().anthropic_oauth_credential_path),
    show_default=True,
    help="Where to store the Anthropic credentials.",
)
@click.option(
    "--token",
    default=None,
    help="Anthropic setup-token value. If provided, skips the OAuth flow.",
)
@click.option(
    "--timeout",
    type=float,
    callback=_validate_oauth_timeout,
    default=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_LOGIN_TIMEOUT_SECONDS,
    show_default=True,
    help="Max seconds allowed for the complete Anthropic OAuth login.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the Anthropic authorization URL automatically.",
)
def configure_anthropic(
    credential_path: str,
    token: str | None,
    timeout: float,
    open_browser: bool,
) -> None:
    """Log in to Anthropic via OAuth (or pass --token to save a setup-token)."""
    if token:
        save_anthropic_setup_token(credential_path, token)
        _print_oauth_login_success("Anthropic", credential_path)
    else:
        _run_anthropic_oauth_login(
            credential_path=credential_path,
            timeout=timeout,
            open_browser=open_browser,
        )

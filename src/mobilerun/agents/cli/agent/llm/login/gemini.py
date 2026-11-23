from typing import Unpack

import click
from mobilerun.cli.base import render

from mobilerun.agents.cli.base.oauth_output import (
    BrowserLoginOptions,
    _print_oauth_login_success,
    _validate_oauth_timeout,
)
from mobilerun.agents.clients.oauth.gemini import run_gemini_oauth_login
from mobilerun.agents.config.settings import get_agent_settings


def _run_gemini_oauth_login(
    credential_path: str, model: str | None, **kwargs: Unpack[BrowserLoginOptions]
) -> None:
    run_gemini_oauth_login(
        credential_path=credential_path,
        model=model,
        on_output=render.message,
        **kwargs,
    )
    _print_oauth_login_success("Gemini", credential_path)


@click.command("gemini")
@click.option(
    "--credential-path",
    default=lambda: str(get_agent_settings().gemini_oauth_credential_path),
    show_default=True,
    help="Where to store Gemini OAuth credentials.",
)
@click.option("--model", default=None, help="Optional model override for later API calls.")
@click.option(
    "--timeout",
    type=float,
    callback=_validate_oauth_timeout,
    default=300.0,
    show_default=True,
    help="Max seconds allowed for the complete Gemini OAuth login.",
)
@click.option(
    "--callback-host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the local OAuth callback server.",
)
@click.option(
    "--callback-port",
    type=int,
    default=0,
    show_default=True,
    help="Port to bind the local OAuth callback server. Use 0 for auto.",
)
@click.option(
    "--callback-path",
    default="/oauth2callback",
    show_default=True,
    help="Callback path for the local OAuth server.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the authorization URL automatically.",
)
def configure_gemini(
    credential_path: str,
    model: str | None,
    timeout: float,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    open_browser: bool,
) -> None:
    """Log in to Gemini Code Assist via OAuth and save credentials locally."""
    _run_gemini_oauth_login(
        credential_path=credential_path,
        model=model,
        timeout=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        open_browser=open_browser,
    )

from typing import Unpack

import click
from mobilerun.cli.base import render

from mobilerun.agents.cli.base.oauth_output import (
    BrowserLoginOptions,
    _print_oauth_login_success,
    _validate_oauth_timeout,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    run_openai_oauth_login,
)
from mobilerun.agents.config.settings import get_agent_settings


def _run_openai_oauth_login(
    credential_path: str, model: str | None, **kwargs: Unpack[BrowserLoginOptions]
) -> None:
    run_openai_oauth_login(
        credential_path=credential_path,
        model=model,
        on_output=render.message,
        **kwargs,
    )
    _print_oauth_login_success("OpenAI", credential_path)


@click.command("openai")
@click.option(
    "--credential-path",
    default=lambda: str(get_agent_settings().openai_oauth_credential_path),
    show_default=True,
    help="Where to store OpenAI OAuth credentials.",
)
@click.option("--model", default=None, help="Optional model override for later API calls.")
@click.option(
    "--timeout",
    type=float,
    callback=_validate_oauth_timeout,
    default=lambda: get_agent_settings().MOBILERUN_OPENAI_OAUTH_LOGIN_TIMEOUT_SECONDS,
    show_default=True,
    help="Max seconds allowed for the complete OpenAI OAuth login.",
)
@click.option(
    "--callback-host",
    default=DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    show_default=True,
    help="Host to bind the local OAuth callback server.",
)
@click.option(
    "--callback-port",
    type=int,
    default=lambda: get_agent_settings().MOBILERUN_OPENAI_OAUTH_CALLBACK_PORT,
    show_default=True,
    help="Port to bind the local OAuth callback server.",
)
@click.option(
    "--callback-path",
    default=DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    show_default=True,
    help="Callback path for the local OAuth server.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the authorization URL automatically.",
)
def configure_openai(
    credential_path: str,
    model: str | None,
    timeout: float,
    callback_host: str,
    callback_port: int,
    callback_path: str,
    open_browser: bool,
) -> None:
    """Log in to ChatGPT/OpenAI via OAuth and save credentials locally."""
    _run_openai_oauth_login(
        credential_path=credential_path,
        model=model,
        timeout=timeout,
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        open_browser=open_browser,
    )

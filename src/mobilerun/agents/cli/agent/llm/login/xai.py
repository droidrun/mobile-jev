from typing import Unpack

import click
from mobilerun.cli.base import render

from mobilerun.agents.cli.base.oauth_output import (
    DeviceLoginOptions,
    _print_oauth_login_success,
    _validate_oauth_timeout,
)
from mobilerun.agents.clients.oauth.grok import run_grok_oauth_login
from mobilerun.agents.config.settings import get_agent_settings


def _run_grok_oauth_login(
    credential_path: str, model: str | None = None, **kwargs: Unpack[DeviceLoginOptions]
) -> None:
    run_grok_oauth_login(
        credential_path=credential_path,
        model=model,
        on_output=render.message,
        **kwargs,
    )
    _print_oauth_login_success("XAI", credential_path)


@click.command("xai")
@click.option(
    "--credential-path",
    default=lambda: str(get_agent_settings().grok_oauth_credential_path),
    show_default=True,
    help="Where to store XAI OAuth credentials.",
)
@click.option("--model", default=None, help="Optional model override for later API calls.")
@click.option(
    "--timeout",
    type=float,
    callback=_validate_oauth_timeout,
    default=300.0,
    show_default=True,
    help="Max seconds allowed for the complete xAI OAuth login.",
)
@click.option(
    "--open-browser/--no-browser",
    default=True,
    show_default=True,
    help="Open the xAI authorization URL automatically.",
)
@click.option(
    "--device-code",
    is_flag=True,
    default=False,
    help="Use xAI's device-code flow for SSH or other headless environments.",
)
def configure_xai(
    credential_path: str,
    model: str | None,
    timeout: float,
    open_browser: bool,
    device_code: bool,
) -> None:
    """Log in to XAI with Mobilerun OAuth."""
    _run_grok_oauth_login(
        credential_path=credential_path,
        model=model,
        timeout=timeout,
        open_browser=open_browser,
        device_code=device_code,
    )

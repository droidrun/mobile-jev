import math
from pathlib import Path
from typing import TypedDict

import click
from mobilerun.cli.base import render
from rich.console import Console

console = Console()


class BrowserLoginOptions(TypedDict, total=False):
    timeout: float | None
    callback_host: str
    callback_port: int | None
    callback_path: str
    open_browser: bool


class DeviceLoginOptions(TypedDict, total=False):
    timeout: float | None
    open_browser: bool
    device_code: bool
    no_browser: bool | None


def _validate_oauth_timeout(
    _ctx: click.Context,
    _param: click.Parameter,
    value: float,
) -> float:
    if not math.isfinite(value) or value <= 0:
        raise click.BadParameter("must be a finite number greater than zero")
    return value


def _print_oauth_login_success(provider_label: str, credential_path: str) -> None:
    render.success(f"{provider_label} login succeeded.")
    render.kv([("Credentials saved to", Path(credential_path).expanduser())])

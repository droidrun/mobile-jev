"""Configure provider profiles and choose the agent's default LLM."""

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from mobilerun.cli.base import menu
from pydantic import HttpUrl, ValidationError

from mobilerun_agents.agent.auth import store
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.settings import DEFAULT_MODELS
from mobilerun_agents.config.user_config import MobileConfig, Vision

from .profiles import configure, configure_jev, providers, remove, set_default
from .wizard import (
    PROVIDER_NAMES,
    environment,
    key_variable,
    list_models,
    pick,
    validate_model,
    wizard,
)


def _save(config: MobileConfig, pending: dict[str, str | None]) -> None:
    path = ConfigLoader.get_user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ConfigLoader and the key store replace files atomically. Keep the original
    # config inode (including comments and permissions) until every write succeeds.
    with TemporaryDirectory(dir=path.parent) as scratch, ExitStack() as rollback:
        backup = Path(scratch) / "config"
        if path.exists():
            os.link(path, backup)
        ConfigLoader.save(config)
        if backup.exists():
            rollback.callback(os.replace, backup, path)
        else:
            rollback.callback(path.unlink)
        for name, value in pending.items():
            previous = store.load(name)
            if value is None:
                store.delete(name)
            else:
                store.save_key(name, value)
            if previous is None:
                rollback.callback(store.delete, name)
            else:
                rollback.callback(store.save_key, name, previous.api_key)
        rollback.pop_all()


@click.command()
@click.argument("default_provider", required=False, metavar="[PROVIDER]")
@click.option("--provider", type=click.Choice(PROVIDER_NAMES), help="Add or update this provider.")
@click.option("--jev-only", is_flag=True, help="Use TypeSafe without an LLM delegate.")
@click.option("--model", help="Model ID (defaults to the saved or suggested model).")
@click.option(
    "--default",
    "make_default",
    is_flag=True,
    help="Make --provider the default, or use --default PROVIDER.",
)
@click.option("--api-key-stdin", is_flag=True, help="Read a provider key from piped stdin.")
@click.option("--base-url", help="Provider endpoint URL, or 'default' for its standard endpoint.")
@click.option(
    "--vision", type=click.Choice(["off", "on", "only"]), help="Choose the selector's vision mode."
)
def setup(
    provider: str | None,
    jev_only: bool,
    model: str | None,
    make_default: bool,
    default_provider: str | None,
    api_key_stdin: bool,
    base_url: str | None,
    vision: Vision | None,
) -> None:
    """Manage providers and the default model. Without flags, open the setup menu."""
    original = ConfigLoader.load()
    config = original.model_copy(deep=True)
    roles = providers(config)
    pending: dict[str, str | None] = {}
    if jev_only and provider != "typesafe":
        raise click.UsageError("Use --jev-only with --provider typesafe.")
    if default_provider is not None:
        if not make_default or provider or model or api_key_stdin or base_url or vision is not None:
            raise click.UsageError("Use --default PROVIDER alone to choose a configured provider.")
        if default_provider not in roles:
            raise click.UsageError("This provider is not configured yet. Add it with --provider.")
        set_default(config, roles[default_provider])
    elif provider == "typesafe":
        if model is not None or base_url is not None:
            raise click.UsageError("--model and --base-url are for LLM providers.")
        decision = configure_jev(config, jev_only, make_default)
        if vision is not None:
            decision.vision = vision
    elif provider is not None:
        previous = config.llm_profiles.get(roles.get(provider, ""))
        chosen = (
            model
            if model is not None
            else (
                previous.model
                if previous
                else DEFAULT_MODELS.get(provider) or next(iter(list_models(provider)), None)
            )
        )
        if chosen is None:
            raise click.UsageError("Supply --model; this adapter has no local model registry.")
        validate_model(provider, chosen.strip())
        profile = configure(config, provider, chosen.strip(), make_default)
        if base_url is not None:
            try:
                profile.base_url = None if base_url == "default" else str(HttpUrl(base_url))
            except ValidationError:
                raise click.UsageError("Expected an HTTP(S) base URL or 'default'.") from None
        if vision is not None:
            profile.vision = vision
    elif (
        not menu.is_interactive()
        or make_default
        or api_key_stdin
        or model is not None
        or base_url is not None
        or vision is not None
    ):
        raise click.UsageError(
            "Use --provider NAME, --default PROVIDER, or open setup in a terminal."
        )
    else:
        try:
            wizard(config, pending)
        except (click.Abort, KeyboardInterrupt, EOFError):
            click.echo("Setup cancelled.")
            return
    if provider is not None:
        env = key_variable(provider)
        if api_key_stdin:
            if not env or sys.stdin.isatty():
                raise click.UsageError("Pipe a key to stdin for a provider requiring an API key.")
            key = sys.stdin.readline().strip()
            if not key:
                raise click.UsageError("API key cannot be empty.")
            pending[provider] = key
        elif env and not environment(provider) and not store.load(provider):
            raise click.UsageError(f"Set {env} or use --api-key-stdin.")
    if config == original and not pending:
        click.echo("Setup unchanged.")
        return
    _save(config, pending)
    click.echo("Provider setup saved.")


@click.command()
@click.argument("provider", required=False)
def logout(provider: str | None) -> None:
    """Remove a provider and its saved key. Environment variables are not changed."""
    config = ConfigLoader.load()
    roles = providers(config)
    if provider is None:
        if not menu.is_interactive():
            raise click.UsageError("Supply a provider, or run logout in a terminal.")
        choices = sorted(roles.keys() | set(store.list_providers()))
        if not choices:
            click.echo("No providers to remove.")
            return
        provider = pick("Provider", choices)
    try:
        name = store.provider_name(provider)
    except ValueError:
        raise click.UsageError("Invalid provider name.") from None
    if name in roles:
        remove(config, name)
        _save(config, {name: None})
    else:
        store.delete(name)
    click.echo(f"No saved provider credentials for {name}.")

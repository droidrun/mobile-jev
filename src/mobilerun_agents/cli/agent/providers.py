"""Store provider credentials separately from profiles and setups."""

import os
import sys

import click
from mobilerun.cli.base import menu
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.auth import oauth, store
from mobilerun_agents.config.user_config import PROVIDER_KINDS

from .configuration import PROVIDERS, pick


def _key_variable(provider: str) -> str | None:
    return (
        f"{provider.upper()}_API_KEY"
        if PROVIDER_KINDS[provider] == "decision"
        else MODELS[provider].env_var
    )


def _environment(provider: str) -> str | None:
    env = _key_variable(provider)
    if env and os.environ.get(env):
        return env
    return "GEMINI_API_KEY" if provider == "gemini" and os.environ.get("GEMINI_API_KEY") else None


def _read_secret() -> str:
    keys = KeyBindings()

    @keys.add("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(exception=click.Abort())

    session: PromptSession[str] = PromptSession()
    return session.prompt("API key: ", is_password=True, key_bindings=keys).strip()


@click.group(cls=menu.MenuGroup)
def providers() -> None:
    """Add, list or remove provider credentials."""


@providers.command()
@click.argument("provider", type=click.Choice(PROVIDERS), required=False)
@click.option("--api-key-stdin", is_flag=True, help="Read the API key from piped stdin.")
def add(provider: str | None, api_key_stdin: bool) -> None:
    """Use an environment key, save an API key, or sign in with OpenRouter."""
    if provider is None:
        if not menu.is_interactive() or api_key_stdin:
            raise click.UsageError("Supply a provider.")
        try:
            saved = set(store.list_providers())
            provider = pick(
                "Provider", PROVIDERS, {name: _status(name, saved) for name in PROVIDERS}
            )
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("Cancelled.")
            return
    env = _key_variable(provider)
    if api_key_stdin:
        if env is None or sys.stdin.isatty():
            raise click.UsageError("Pipe a key to stdin for a provider requiring an API key.")
        key = sys.stdin.readline().strip()
    elif env is None:
        click.echo(f"{provider} does not require an API key.")
        return
    elif not menu.is_interactive():
        if _environment(provider) or provider in store.list_providers():
            click.echo(f"Credentials available for {provider}.")
            return
        raise click.UsageError(f"Set {env} or use --api-key-stdin.")
    else:
        methods = ["API key"]
        if provider == "openrouter":
            methods.append("Sign in with browser")
        if _environment(provider) or provider in store.list_providers():
            methods.insert(0, "Use existing key")
        try:
            method = pick("Authentication", methods)
            match method:
                case "Use existing key":
                    click.echo(f"Credentials available for {provider}.")
                    return
                case "Sign in with browser":
                    key = oauth.browser_key()
                case _:
                    key = _read_secret()
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("Cancelled.")
            return
        except oauth.OAuthError:
            raise click.ClickException("OpenRouter browser sign-in failed.") from None
    if not key:
        raise click.UsageError("API key cannot be empty.")
    store.save_key(provider, key)
    click.echo(f"API key saved for {provider}.")
    if existing_env := _environment(provider):
        click.echo(f"{existing_env} takes precedence during runs.")


@providers.command("list")
def list_providers() -> None:
    """List available credential sources without displaying key values."""
    saved = set(store.list_providers())
    for name in PROVIDERS:
        status = _status(name, saved)
        if status != "no key":
            click.echo(f"{name}\t{status}")


def _status(name: str, saved: set[str]) -> str:
    if env := _environment(name):
        return f"{env} set"
    if name in saved:
        return "key saved"
    return "no key" if _key_variable(name) else "local (no key required)"


@providers.command()
@click.argument("provider", type=click.Choice(PROVIDERS))
def remove(provider: str) -> None:
    """Remove the saved key; environment variables are not changed."""
    store.delete(provider)
    click.echo(f"No saved provider credentials for {provider}.")

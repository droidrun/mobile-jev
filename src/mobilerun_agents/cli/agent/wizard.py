"""Interactive provider menu. No local writes until Done."""

import os
import re
from typing import Any

import click
from llama_index.llms.anthropic.utils import ANTHROPIC_MODELS
from llama_index.llms.google_genai.base import DEFAULT_MODEL as GEMINI_DEFAULT
from llama_index.llms.openai.utils import (
    GPT4_MODELS,
    O1_MODELS,
    RESPONSES_API_ONLY_MODELS,
    TURBO_MODELS,
)
from mobilerun.cli.base import prompts
from mobilerun.cli.base.prompts import SelectChoice
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.auth import oauth, store
from mobilerun_agents.config.user_config import MobileConfig

from .profiles import (
    configure,
    configure_jev,
    default_role,
    primary,
    providers,
    remove,
    set_default,
)

PROVIDER_NAMES = sorted(MODELS.keys() | {"typesafe"})


def key_variable(provider: str) -> str | None:
    return "TYPESAFE_API_KEY" if provider == "typesafe" else MODELS[provider].env_var


def list_models(provider: str) -> list[str]:
    """Installed registry only. Adapters do not publish release timestamps."""
    if provider == "openai":
        names = (
            O1_MODELS.keys()
            | GPT4_MODELS.keys()
            | TURBO_MODELS.keys()
            | RESPONSES_API_ONLY_MODELS.keys()
        )
    elif provider == "anthropic":
        names = set(ANTHROPIC_MODELS)
    elif provider == "gemini":
        return [GEMINI_DEFAULT]
    else:
        return []

    def generation(name: str) -> tuple[bool, tuple[int, ...], str]:
        version = re.search(r"\d+(?:[.-]\d+)*", name)
        numbers = tuple(map(int, re.split(r"[.-]", version[0]))) if version else ()
        return name.startswith(("gpt-", "claude-")), numbers, name

    return sorted(names, key=generation, reverse=True)


def pick(title: str, labels: list[str], descriptions: dict[str, str] | None = None) -> str:
    value = prompts.select_prompt(
        title, [SelectChoice(s, descriptions[s] if descriptions else s) for s in labels]
    )
    if value is None:
        raise click.Abort()
    return value


def validate_model(provider: str, model: str) -> None:
    # Only these native adapters read a local registry. Gemini and Ollama
    # can query a server for metadata; never probe them during setup.
    if provider not in {"openai", "anthropic"}:
        return
    try:
        options: dict[str, Any] = {"model": model, "api_key": "offline-validation"}
        _ = MODELS[provider].adapter(**options).metadata
    except ValueError:
        raise click.UsageError(
            f"The installed {provider} adapter does not recognize this model. "
            "Choose a listed model, or use --provider openai_like --base-url URL "
            "--model ID for a custom OpenAI-compatible endpoint."
        ) from None


def read_secret() -> str:
    # Base text_prompt conflates Escape with an empty Enter. Only Escape cancels.
    keys = KeyBindings()

    @keys.add("escape")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(exception=click.Abort())

    session: PromptSession[str] = PromptSession()
    return session.prompt("API key: ", is_password=True, key_bindings=keys).strip()


def environment(provider: str) -> str | None:
    env = key_variable(provider)
    if env and os.environ.get(env):
        return env
    return "GEMINI_API_KEY" if provider == "gemini" and os.environ.get("GEMINI_API_KEY") else None


def _authenticate(provider: str, pending: dict[str, str | None], change: bool) -> None:
    if not key_variable(provider):
        if not change:
            pick("Authentication", ["None (local)"])
        return
    env = environment(provider)
    saved = store.load(provider) if provider not in pending else None
    existing = pending.get(provider) or (saved.api_key if saved else None)
    if change and (env or existing):
        return
    methods = ["API key"]
    if provider == "openrouter":
        methods.append("Sign in with browser")
    if env:
        methods.insert(0, f"Use environment variable {env}")
    if existing:
        methods.insert(0, "Use saved API key")
    method = pick("Authentication", methods)
    if env and method == f"Use environment variable {env}":
        return
    if method == "Use saved API key":
        return
    if method == "Sign in with browser":
        try:
            key = oauth.browser_key()
        except oauth.OAuthError as exc:
            raise click.ClickException(str(exc)) from None
    else:
        while not (key := read_secret()):
            click.echo("Key is empty. Enter a key, or press Esc to cancel.")
    pending[provider] = key
    if env:
        click.echo(f"{env} takes precedence during runs; unset it to use the saved key.")


def wizard(config: MobileConfig, pending: dict[str, str | None]) -> None:
    while True:
        roles = providers(config)
        click.echo("Configured providers:" if roles else "No providers configured.")
        saved = set(store.list_providers())
        descriptions = {}
        for provider in PROVIDER_NAMES:
            env = environment(provider)
            stored = bool(pending[provider]) if provider in pending else provider in saved
            if env:
                status = f"✓ {env} set"
            elif stored:
                status = "✓ key saved"
            else:
                status = "no key" if key_variable(provider) else "local"
            label = "typesafe (Jev decision model)" if provider == "typesafe" else provider
            descriptions[provider] = f"{label:<30} {status}"
            if provider in roles:
                role = roles[provider]
                default = " default" if role == primary(config) else ""
                model = config.llm_profiles[role].model if role in config.llm_profiles else ""
                click.echo(f"  {descriptions[provider]}{default}  {model}".rstrip())
        available_names = sorted(set(PROVIDER_NAMES) - roles.keys())
        actions = (
            (["Add provider"] if available_names else [])
            + (["Change model", "Set default", "Remove provider"] if roles else [])
            + ["Done"]
        )
        action = pick("Setup", actions)
        if action == "Done":
            return
        adding = action == "Add provider"
        provider = pick("Provider", available_names if adding else sorted(roles), descriptions)
        if action == "Set default":
            set_default(config, roles[provider])
        elif action == "Remove provider":
            remove(config, provider)
            pending[provider] = None
        else:
            staged = dict(pending)
            _authenticate(provider, staged, change=not adding)
            if provider == "typesafe":
                llm = config.llm_profiles.get(default_role(config) or "")
                target = f"{llm.provider}/{llm.model}" if llm else "the default LLM"
                mode = pick(
                    "How should Jev run?", [f"Jev + delegate to {target} (recommended)", "Jev only"]
                )
            try:
                if provider == "typesafe":
                    configure_jev(config, mode == "Jev only", True)
                else:
                    models = list_models(provider)
                    model = pick("Model", [*models, "other…"])
                    if model == "other…":
                        model = prompts.text_prompt("Model identifier")
                        if not model:
                            raise click.Abort()
                    validate_model(provider, model)
                    configure(config, provider, model, False)
            except click.UsageError as exc:
                click.echo(exc.format_message())
                continue
            pending.update(staged)
            if provider == "typesafe":
                continue

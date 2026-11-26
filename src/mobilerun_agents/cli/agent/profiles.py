"""Manage named model profiles independently of provider credentials."""

import asyncio
from typing import cast

import click
from llama_index.llms.anthropic.utils import ANTHROPIC_MODELS
from llama_index.llms.google_genai.base import DEFAULT_MODEL as GEMINI_DEFAULT
from llama_index.llms.openai.utils import (
    GPT4_MODELS,
    O1_MODELS,
    RESPONSES_API_ONLY_MODELS,
    TURBO_MODELS,
)
from mobilerun.cli.base import menu, prompts
from pydantic import ValidationError

from mobilerun_agents.agent.action_selector.llm.factory import build_llm
from mobilerun_agents.config.user_config import (
    PROVIDER_KINDS,
    DecisionProfile,
    LLMProfile,
    Reasoning,
    Vision,
    profile_validation_message,
)
from mobilerun_agents.errors import UnknownModel

from .configuration import PROVIDERS, VISION_MODES, load_config, pick, save_config

REASONING_LEVELS = ["low", "medium", "high"]


def _pick_model(provider: str) -> str:
    match provider:
        case "openai":
            names = (
                GPT4_MODELS.keys()
                | O1_MODELS.keys()
                | RESPONSES_API_ONLY_MODELS.keys()
                | TURBO_MODELS.keys()
            )
        case "anthropic":
            names = set(ANTHROPIC_MODELS)
        case "gemini":
            names = {GEMINI_DEFAULT}
        case _:
            names = set()
    model = pick("Model", [*sorted(names, reverse=True), "other…"])
    if model == "other…":
        model = prompts.text_prompt("Model ID")
    if not model.strip():
        raise click.Abort()
    return model


@click.group(cls=menu.MenuGroup)
def profiles() -> None:
    """Add, list or remove named model profiles."""


@profiles.command()
@click.argument("name", required=False)
@click.option("--provider", type=click.Choice(PROVIDERS))
@click.option("--model", help="Model ID (required for LLM profiles).")
@click.option("--reasoning", type=click.Choice(REASONING_LEVELS))
@click.option("--vision", type=click.Choice(VISION_MODES))
@click.option("--base-url", help="HTTP(S) endpoint without userinfo, query, or fragment.")
@click.option(
    "--min-confidence", type=click.FloatRange(0, 1), help="Decision model confidence threshold."
)
def add(
    name: str | None,
    provider: str | None,
    model: str | None,
    reasoning: Reasoning | None,
    vision: Vision | None,
    base_url: str | None,
    min_confidence: float | None,
) -> None:
    """Add or replace NAME."""
    config = load_config()
    if menu.is_interactive() and (
        name is None or provider is None or (PROVIDER_KINDS[provider] == "llm" and model is None)
    ):
        try:
            if name is None:
                name = prompts.text_prompt("Profile name")
            if not name.strip():
                raise click.Abort()
            provider = provider or pick("Provider", PROVIDERS)
            decision = PROVIDER_KINDS[provider] == "decision"
            if not decision and model is None:
                model = _pick_model(provider)
            if vision is None:
                selected_vision = pick("Vision", ["Default", *VISION_MODES])
                vision = cast(Vision, selected_vision) if selected_vision != "Default" else None
            if decision and min_confidence is None:
                confidence = prompts.text_prompt("Minimum confidence (optional)")
                if confidence:
                    try:
                        min_confidence = float(confidence)
                    except ValueError:
                        raise click.UsageError(
                            "Minimum confidence must be between 0 and 1."
                        ) from None
            elif not decision and reasoning is None:
                selected_reasoning = pick("Reasoning", ["Default", *REASONING_LEVELS])
                reasoning = (
                    cast(Reasoning, selected_reasoning) if selected_reasoning != "Default" else None
                )
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("Cancelled.")
            return
    if not name or not name.strip() or provider is None:
        raise click.UsageError("Supply NAME and --provider.")
    try:
        if PROVIDER_KINDS[provider] == "decision":
            if model is not None or reasoning is not None or base_url is not None:
                raise click.UsageError(
                    "Decision model profiles reject --model, --reasoning and --base-url."
                )
            config.profiles[name] = DecisionProfile(
                provider=provider, vision=vision, min_confidence=min_confidence
            )
        else:
            if min_confidence is not None:
                raise click.UsageError("--min-confidence requires a decision model profile.")
            if not model or not (model := model.strip()):
                raise click.UsageError("--model is required for LLM profiles.")
            if provider in {"openai", "anthropic"}:
                # These native adapters use a local model registry. Do not
                # call providers whose metadata can trigger a network request.
                asyncio.run(build_llm(provider, model, api_key="offline-validation"))
            config.profiles[name] = LLMProfile(
                provider=provider,
                model=model,
                reasoning=reasoning,
                vision=vision,
                base_url=base_url,
            )
    except UnknownModel:
        raise click.UsageError(
            f"The installed {provider} adapter does not recognize this model. "
            "Use --provider openai_like --base-url URL --model ID for a custom endpoint."
        ) from None
    except ValidationError as exc:
        if message := profile_validation_message(exc):
            raise click.ClickException(message) from None
        field = "HTTP(S) base URL" if base_url is not None else "profile settings"
        raise click.UsageError(f"Invalid {field}.") from None
    save_config(config)
    click.echo(f"Profile saved: {name}")


@profiles.command("list")
def list_profiles() -> None:
    """List saved profiles."""
    for name, profile in load_config().profiles.items():
        match profile:
            case LLMProfile(provider=provider, model=model):
                click.echo(f"{name}\t{provider}/{model}")
            case DecisionProfile(provider=provider):
                click.echo(f"{name}\t{provider} (decision model)")


@profiles.command()
@click.argument("name")
def remove(name: str) -> None:
    """Remove NAME; first remove or update setups that reference it."""
    config = load_config()
    if name not in config.profiles:
        raise click.UsageError(f"Unknown profile {name!r}")
    del config.profiles[name]
    save_config(config)
    click.echo(f"Profile removed: {name}")

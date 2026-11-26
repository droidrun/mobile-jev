"""Shared presentation and persistence for agent configuration commands."""

import click
from mobilerun.cli.base import prompts
from mobilerun.cli.base.prompts import SelectChoice
from pydantic import ValidationError

from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import (
    PROVIDER_KINDS,
    MobileConfig,
    profile_validation_message,
)

PROVIDERS = sorted(PROVIDER_KINDS)
VISION_MODES = ["off", "on", "only"]


def pick(title: str, labels: list[str], details: dict[str, str] | None = None) -> str:
    choices = [SelectChoice(label, label, (details or {}).get(label)) for label in labels]
    return pick_choice(title, choices)


def pick_choice(title: str, choices: list[SelectChoice]) -> str:
    selected = prompts.select_prompt(title, choices)
    if selected is None:
        raise click.Abort()
    return selected


def load_config(path: str | None = None) -> MobileConfig:
    try:
        return ConfigLoader.load(path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None


def save_config(config: MobileConfig) -> None:
    try:
        ConfigLoader.save(config)
    except ValidationError as exc:
        raise click.UsageError(profile_validation_message(exc) or str(exc)) from None
    except OSError:
        raise click.ClickException("Could not save agents.yaml.") from None

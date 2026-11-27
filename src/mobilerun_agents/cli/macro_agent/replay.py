"""Reuse base replay's device selection, credentials, limits and dry-run mode."""

from typing import Unpack

import click
from mobilerun.macro.cli.replay import (
    ReplayOptions,
    _preflight_before_target,
    replay_options,
    run_replay,
)

from mobilerun_agents.cli.agent.configuration import load_config
from mobilerun_agents.config.user_config import LLMProfile
from mobilerun_agents.macro_agent import MacroAgentOperator


@click.command()
@click.option("--provider", help="LLM provider for recovery.")
@click.option("--model", help="LLM model for recovery.")
@click.option("--yes", "-y", is_flag=True, help="Confirm possible SMS charges without prompting.")
@_preflight_before_target
@replay_options
def replay(provider: str | None, model: str | None, **options: Unpack[ReplayOptions]) -> None:
    """Replay a macro by name, folder, or version. Repair needs a recorded after fingerprint."""
    replay_handoff(provider, model, **options)


def replay_handoff(
    provider: str | None, model: str | None, **options: Unpack[ReplayOptions]
) -> None:
    """Also used by base's `macro replay --on-mismatch agent` plugin hook."""
    if bool(provider) != bool(model):
        raise click.UsageError("Provide both --provider and --model, or neither")
    if options["dry_run"]:
        run_replay(MacroAgentOperator, **options)
        return
    config = load_config()
    if provider and model:
        config = config.model_copy(deep=True)
        config.default_setup = None
        config.setups = {}
        config.profiles = {"recovery": LLMProfile(provider=provider, model=model)}
    run_replay(MacroAgentOperator, config=config, **options)

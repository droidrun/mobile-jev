"""Reuse base replay's device selection, credentials, limits and dry-run mode."""

from os import PathLike, fsdecode
from typing import Unpack

import click
from mobilerun.macro.cli.replay import ReplayOptions, replay_options, run_replay

from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import LLMProfile
from mobilerun_agents.macro_agent import MacroAgentOperator
from mobilerun_agents.macro_agent.paths import resolve_macro_target


class _ReplayTarget(click.Path):
    def convert(
        self, value: str | PathLike[str], param: click.Parameter | None, ctx: click.Context | None
    ) -> str:
        return fsdecode(super().convert(str(resolve_macro_target(value)), param, ctx))


@click.command()
@click.option("--provider", help="LLM provider for recovery.")
@click.option("--model", help="LLM model for recovery.")
@replay_options
def replay(provider: str | None, model: str | None, **options: Unpack[ReplayOptions]) -> None:
    """Replay a run ID, macro folder, or macro.json; hand off on UI divergence."""
    replay_handoff(provider, model, **options)


# Keep base's replay flags and device selection; resolve IDs before checking path existence.
next(param for param in replay.params if param.name == "path").type = _ReplayTarget(exists=True)


def replay_handoff(
    provider: str | None, model: str | None, **options: Unpack[ReplayOptions]
) -> None:
    """Also used by base's `macro replay --on-mismatch agent` plugin hook."""
    if bool(provider) != bool(model):
        raise click.UsageError("Provide both --provider and --model, or neither")
    config = ConfigLoader.load()
    if provider and model:
        config = config.model_copy(deep=True)
        config.agent.delegation = None
        config.agent.selector = "default"
        config.default_llm = "default"
        config.decision_profiles = {}
        config.llm_profiles = {"default": LLMProfile(provider=provider, model=model)}
    run_replay(MacroAgentOperator, config=config, **options)

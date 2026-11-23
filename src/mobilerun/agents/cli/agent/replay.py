"""Replay an agent macro, resuming with the agent on divergence."""

from typing import TypedDict, Unpack

import click
from mobilerun.cli.devices.base.options import DeviceTarget
from mobilerun.macro.cli.replay import replay_options, run_replay

from mobilerun.agents.config.loader import ConfigLoader
from mobilerun.agents.config.user_config import LLMProfile
from mobilerun.agents.operators import AgentMacroOperator


class ReplayOptions(TypedDict):
    path: str
    target: DeviceTarget
    delay: float
    start_from: int
    max_steps: int | None
    debug: bool
    dry_run: bool
    state_timeout: float
    state_threshold: float
    config_path: str | None


@click.command(short_help="Replay with agent recovery")
@click.option("--provider", default=None, help="LLM provider override for the agent macro")
@click.option("--model", default=None, help="LLM model override for the agent macro")
@replay_options
def replay(provider: str | None, model: str | None, **options: Unpack[ReplayOptions]) -> None:
    """Replay FILE on DEVICE; let the agent finish if UI state diverges."""
    if bool(provider) != bool(model):
        raise click.UsageError("Provide both --provider and --model, or neither")
    config = ConfigLoader.load(options["config_path"])
    if provider and model:
        config.llm_profiles = {
            name: LLMProfile(provider=provider, model=model) for name in config.llm_profiles
        }
    run_replay(AgentMacroOperator, config=config, **options)

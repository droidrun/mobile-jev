"""Run the selected workflow on a device."""

import asyncio
from functools import wraps
from typing import Any

import click
from mobilerun.cli.base.safety import confirm_action
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.config.settings import get_settings
from mobilerun.errors import CloudAuthRequired
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun_agents.agent.action_selector.loader import build_setup
from mobilerun_agents.agent.auth import CredentialMissing
from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.operators import AgentOperator, PlanningOperator
from mobilerun_agents.cli.agent.configuration import VISION_MODES, load_config
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import Vision
from mobilerun_agents.errors import UnknownModel


@device_command(allow_auto_setup=True)
def _device_target(target: DeviceTarget) -> DeviceTarget:
    # Resolve the model before touching the device.
    return target


@click.command()
@click.argument("goal")
@click.option("--config", "agent_config", type=click.Path(exists=True, dir_okay=False))
@click.option("--setup", help="Use this named setup instead of the default.")
@click.option("--steps", type=click.IntRange(min=1))
@click.option(
    "--vision",
    type=click.Choice(VISION_MODES),
    help="Override selector vision for this run.",
)
@click.option("--yes", "-y", is_flag=True, help="Confirm cloud device charges.")
@wraps(_device_target, assigned=())
def run(
    goal: str,
    agent_config: str | None,
    setup: str | None,
    steps: int | None,
    vision: Vision | None,
    yes: bool,
    **device_options: Any,
) -> None:
    """Run GOAL on a mobile device."""
    configure_cli_logging(debug=bool(click.get_current_context().find_root().params.get("debug")))
    try:
        config = load_config(agent_config)
        with ConfigLoader.override(config), asyncio.Runner() as loop:
            try:
                spec = loop.run(build_setup(setup, vision))
                operator = (
                    PlanningOperator(planner=spec.planner, executor=spec.executor)
                    if spec.planner is not None
                    else AgentOperator(selector=spec.selector)
                )
            except (CredentialMissing, UnknownModel, ValueError) as exc:
                raise click.ClickException(str(exc)) from None
            target = _device_target(**device_options)
            if target.kind == "cloud":
                confirm_action(
                    "run a cloud task",
                    get_settings().MOBILERUN_BILLING.value,
                    f"Using {target} may incur charges.",
                    yes=yes,
                )
            try:
                loop.run(_execute(operator, target, goal, steps))
            except CloudAuthRequired as exc:
                raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        raise click.exceptions.Exit(130) from None


async def _execute(
    operator: AgentOperator | PlanningOperator, target: DeviceTarget, goal: str, steps: int | None
) -> None:
    async with target.open() as device:
        if device is None:
            raise click.UsageError("A device is required.")
        result: ExecutionResult[BaseModel] = await operator.execute(device, goal, max_steps=steps)
    if not result.success:
        raise click.exceptions.Exit(1)

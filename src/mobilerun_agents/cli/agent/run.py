"""Run the selected workflow on a device."""

import asyncio
from functools import wraps
from typing import Any

import click
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.core.operator import Operator
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
@click.option("--no-record", is_flag=True, help="Do not save a recording for this run.")
@click.option(
    "--vision",
    type=click.Choice(VISION_MODES),
    help="Override selector vision for this run.",
)
@wraps(_device_target, assigned=())
def run(
    goal: str,
    agent_config: str | None,
    setup: str | None,
    steps: int | None,
    no_record: bool,
    vision: Vision | None,
    **device_options: Any,
) -> None:
    """Run GOAL on a mobile device."""
    configure_cli_logging(debug=bool(click.get_current_context().find_root().params.get("debug")))
    try:
        config = load_config(agent_config)
        with ConfigLoader.override(config), asyncio.Runner() as loop:
            try:
                spec = loop.run(build_setup(setup, vision))
                record = False if no_record else None
                operator = (
                    PlanningOperator(planner=spec.planner, executor=spec.executor, record=record)
                    if spec.planner is not None
                    else AgentOperator(selector=spec.selector, record=record)
                )
            except (CredentialMissing, UnknownModel, ValueError) as exc:
                raise click.ClickException(str(exc)) from None
            target = _device_target(**device_options)
            try:
                loop.run(_execute(operator, target, goal, steps))
            except CloudAuthRequired as exc:
                raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        raise click.exceptions.Exit(130) from None


async def _execute(operator: Operator, target: DeviceTarget, goal: str, steps: int | None) -> None:
    async with target.open() as device:
        if device is None:
            raise click.UsageError("A device is required.")
        result: ExecutionResult[BaseModel] = await operator.execute(device, goal, max_steps=steps)
    if not result.success:
        raise click.exceptions.Exit(1)

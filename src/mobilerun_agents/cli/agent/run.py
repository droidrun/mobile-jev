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

from mobilerun_agents.agent.action_selector.llm.factory import MODELS
from mobilerun_agents.agent.action_selector.loader import (
    NoModelConfigured,
    default_planner,
    default_selector,
)
from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.operators import AgentOperator, PlanningOperator
from mobilerun_agents.cli.agent.profiles import default_role, primary, providers
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import Vision


@device_command(allow_auto_setup=True)
def _device_target(target: DeviceTarget) -> DeviceTarget:
    # Resolve the model before touching the device.
    return target


@click.command()
@click.argument("goal")
@click.option("--config", "agent_config", type=click.Path(exists=True, dir_okay=False))
@click.option("-p", "--provider", type=click.Choice(sorted(MODELS)))
@click.option("-m", "--model")
@click.option("--plan", is_flag=True, help="Use a separate text-only LLM planner.")
@click.option("--steps", type=click.IntRange(min=1))
@click.option(
    "--vision",
    type=click.Choice(["off", "on", "only"]),
    help="Override selector vision for this run.",
)
@click.option("--no-delegate", is_flag=True)
@click.option("--yes", "-y", is_flag=True, help="Confirm cloud device charges.")
@wraps(_device_target, assigned=())
def run(
    goal: str,
    agent_config: str | None,
    provider: str | None,
    model: str | None,
    plan: bool,
    steps: int | None,
    vision: Vision | None,
    no_delegate: bool,
    yes: bool,
    **device_options: Any,
) -> None:
    """Run GOAL on a mobile device."""
    configure_cli_logging(debug=bool(click.get_current_context().find_root().params.get("debug")))
    config = ConfigLoader.load(agent_config)
    if no_delegate:
        config.agent.delegation = None
    if provider or model:
        role = (default_role(config) or primary(config)) if plan else primary(config)
        if role == "default" and role not in config.llm_profiles and config.default_llm:
            role = config.default_llm
        if provider:
            selected = providers(config).get(provider)
            if selected is None:
                raise click.UsageError(f"Provider {provider!r} is not configured.")
            if config.agent.delegation and config.agent.delegation.primary:
                config.agent.delegation.primary = selected
            else:
                config.agent.selector = selected
            role = selected
            if plan:
                config.default_llm = selected
        profile = config.llm_profiles.get(role)
        if profile is None:
            raise click.UsageError("Model overrides require an LLM selector profile.")
        if model:
            profile.model = model
    with ConfigLoader.override(config), asyncio.Runner() as loop:
        try:
            operator = (
                PlanningOperator(
                    planner=loop.run(default_planner()),
                    executor=loop.run(default_selector("executor", vision)),
                )
                if plan
                else AgentOperator(selector=loop.run(default_selector("agent", vision)))
            )
        except NoModelConfigured as exc:
            raise click.ClickException(str(exc)) from exc
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


async def _execute(
    operator: AgentOperator | PlanningOperator, target: DeviceTarget, goal: str, steps: int | None
) -> None:
    async with target.open() as device:
        if device is None:
            raise click.UsageError("A device is required.")
        result: ExecutionResult[BaseModel] = await operator.execute(device, goal, max_steps=steps)
    if not result.success:
        raise click.exceptions.Exit(1)

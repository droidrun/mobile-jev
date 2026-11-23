"""Run an agent from the command line."""

import sys

import click
from mobilerun.cli.base.logging import remember_cli_handler
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.cli.devices.base.options import DeviceTarget, device_command
from mobilerun.config.settings import get_settings
from mobilerun.logs.handlers import LogLevel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.logs.cli import _setup_cli_logging
from mobilerun.agents.runner import run_agent
from mobilerun.agents.telemetry import print_telemetry_message
from mobilerun.agents.workflows.external import list_agents

try:
    _available_agents = list_agents()
except Exception:
    _available_agents = []


def _show_local_config(config: MobileConfig, debug: bool, level: LogLevel | None = None) -> None:
    if level in ("verbose", "debug"):
        console = Console()
        if config.logging.rich_text:
            cloud_text = Text()
            cloud_text.append("✨ Try Mobilerun Cloud: ", style="bold cyan")
            cloud_text.append("https://cloud.mobilerun.ai/sign-in", style="bold blue underline")
            console.print(Panel(cloud_text, border_style="cyan", padding=(0, 1)))
        else:
            console.print("\n✨ Try Mobilerun Cloud: https://cloud.mobilerun.ai/sign-in\n")
    handler = _setup_cli_logging(debug, level)
    remember_cli_handler(handler)
    if level != "quiet":
        print_telemetry_message(config_enabled=config.telemetry.enabled)


@click.command()
@click.argument("command", type=str)
@click.option("--config", help="Path to config file for agent/LLM settings", default=None)
@click.option(
    "--agent",
    type=click.Choice(_available_agents) if _available_agents else None,
    help="External agent to use"
    + (f" [{', '.join(_available_agents)}]" if _available_agents else " (none available)"),
    default=None,
)
@click.option(
    "--provider",
    "-p",
    help="LLM provider (OpenAI, openai_oauth, XAI, Ollama, Anthropic, anthropic_oauth, GoogleGenAI, gemini_oauth_code_assist, DeepSeek)",
    default=None,
)
@click.option(
    "--model",
    "-m",
    help="LLM model name",
    default=None,
)
@click.option("--temperature", type=float, help="Temperature for LLM", default=None)
@click.option(
    "--steps",
    type=int,
    help="Maximum number of agent steps (local and cloud; default from config agent.max_steps, 15)",
    default=None,
)
@click.option(
    "--llm-base-url",
    "--base_url",
    "-u",
    "base_url",
    help="LLM provider base_url (e.g., OpenRouter or Ollama); overrides its config value.",
    default=None,
)
@click.option(
    "--llm-api-base",
    "--api_base",
    "api_base",
    help="LLM provider api_base (e.g., OpenAI-like); overrides its config value. Provider decides if both URL settings are supplied.",
    default=None,
)
@click.option(
    "--vision/--no-vision",
    default=None,
    help="Enable vision capabilites by using screenshots for all agents.",
)
@click.option(
    "--vision-only/--no-vision-only",
    default=None,
    help="Use screenshots only without an accessibility tree.",
)
@click.option("--reasoning/--no-reasoning", default=None, help="Enable planning with reasoning")
@click.option(
    "--stream/--no-stream",
    default=None,
    help="Stream LLM responses to console in real-time",
)
@click.option("--tracing/--no-tracing", default=None, help="Enable Arize Phoenix tracing")
@click.option("--debug/--no-debug", default=None, help="Enable verbose debug logging")
@click.option("-q", "--quiet", is_flag=True, help="Show only results and warnings.")
@click.option("--verbose", is_flag=True, help="Show reasoning and tool output.")
@click.option("--yes", "-y", is_flag=True, help="Confirm cloud task charges.")
@click.option(
    "--save-trajectory",
    type=click.Choice(["none", "step", "action"]),
    help="Trajectory saving level: none (no saving), step (save per step), action (save per action)",
    default=None,
)
@device_command(base_url_name="device_base_url", allow_auto_setup=True)
@coro
async def run(
    command: str,
    target: DeviceTarget,
    config: str | None,
    agent: str | None,
    provider: str | None,
    model: str | None,
    steps: int | None,
    base_url: str | None,
    api_base: str | None,
    temperature: float | None,
    vision: bool | None,
    vision_only: bool | None,
    reasoning: bool | None,
    stream: bool | None,
    tracing: bool | None,
    debug: bool | None,
    quiet: bool,
    verbose: bool,
    yes: bool,
    save_trajectory: str | None,
) -> None:
    """Run a command on your mobile device using natural language.

    Example: mobilerun agent run "Open settings" DEVICE
    """

    if sum((quiet, verbose, debug is not None)) > 1:
        raise click.UsageError("--quiet, --verbose, and --debug/--no-debug are mutually exclusive")
    cli_level: LogLevel | None = (
        "quiet"
        if quiet
        else (
            "verbose"
            if verbose
            else "debug" if debug is True else "normal" if debug is False else None
        )
    )

    def show_config(config: MobileConfig, debug_mode: bool) -> None:
        _show_local_config(config, debug_mode, config.logging.level)

    kind = target.kind

    if kind == "cloud":
        await confirm_action_async(
            "run a cloud task",
            get_settings().MOBILERUN_BILLING.value,
            f"Running this task on cloud device {target} may incur charges.",
            yes=yes,
        )

    async with target.open() as device:
        result = await run_agent(
            command=command,
            device=device,
            config_path=config,
            target=target.identifier,
            device_kind=kind,
            agent=agent,
            provider=provider,
            model=model,
            steps=steps,
            base_url=base_url,
            api_base=api_base,
            vision=vision,
            vision_only=vision_only,
            reasoning=reasoning,
            stream=stream,
            tracing=tracing,
            debug=debug,
            log_level=cli_level,
            tcp=target.tcp,
            device_base_url=target.base_url,
            temperature=temperature,
            save_trajectory=save_trajectory,
            keyboard_cleanup_failure_handler=click.echo,
            on_config=show_config,
            on_cloud_warning=click.echo,
            on_cloud_task_started=lambda task_id: click.echo(f"Cloud task_id={task_id}"),
        )

    if result.error:
        click.echo(result.error)
    if result.cloud_result is not None:
        cloud_result = result.cloud_result
        click.echo(f"Answer: {cloud_result.answer or ''}")
        click.echo(f"Success: {cloud_result.success}")
        click.echo(f"Steps: {cloud_result.steps}")
        if cloud_result.output is not None:
            click.echo(f"Structured output: {cloud_result.output}")
    sys.exit(0 if result.success else 1)

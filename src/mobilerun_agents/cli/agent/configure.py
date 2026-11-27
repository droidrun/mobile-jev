"""Compose named profiles into a straight or planning setup."""

import asyncio

import click
import yaml
from mobilerun.cli.base import menu, prompts
from mobilerun.cli.base.prompts import SelectChoice
from pydantic import ValidationError

from mobilerun_agents.agent.auth import CredentialMissing, resolve
from mobilerun_agents.config.user_config import (
    DecisionProfile,
    LLMProfile,
    MobileConfig,
    SetupConfig,
)

from .configuration import load_config, pick, pick_choice, update_config


class _SetupGroup(click.Group):
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return super().get_command(ctx, cmd_name) or (
            None if cmd_name.startswith("-") else _edit_setup
        )

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        # `default NAME` sets the default; `default --selector …` edits a
        # setup named "default", the name suggested by the wizard.
        role_flags = {"--selector", "--planner", "--executor", "--delegate"}
        if (
            args
            and args[0] == "default"
            and any(a.partition("=")[0] in role_flags for a in args[1:])
        ):
            return args[0], _edit_setup, args[1:]
        return super().resolve_command(ctx, args)


@click.group(cls=_SetupGroup, invoke_without_command=True, subcommand_metavar="[NAME | COMMAND]")
@click.pass_context
def configure(ctx: click.Context) -> None:
    """Compose named profiles into a setup.

    \b
    configure [NAME]                                    Interactive menu
    configure NAME --selector PROFILE [--delegate PROFILE]
    configure NAME --planner PROFILE --executor PROFILE [--delegate PROFILE]
    """
    if ctx.invoked_subcommand is None:
        _interactive(None)


@click.command()
@click.option("--selector", help="Profile for a straight setup.")
@click.option("--planner", help="LLM profile for a planning setup.")
@click.option("--executor", help="Action profile for a planning setup.")
@click.option("--delegate", help="LLM fallback for a decision model selector or executor.")
@click.pass_context
def _edit_setup(
    ctx: click.Context,
    selector: str | None,
    planner: str | None,
    executor: str | None,
    delegate: str | None,
) -> None:
    """Add or replace this setup."""
    name = ctx.info_name or ""
    if all(role is None for role in (selector, planner, executor, delegate)):
        _interactive(name)
        return
    try:
        setup = SetupConfig(
            selector=selector, planner=planner, executor=executor, delegate=delegate
        )
    except ValidationError as exc:
        raise click.UsageError(str(exc)) from None
    _save_setup(name, setup)


def _save_setup(
    name: str,
    setup: SetupConfig,
    make_default: bool = False,
    clear_default: bool = False,
) -> None:
    if not name.strip() or name in {"show", "remove"} or name.startswith("-"):
        raise click.UsageError("Choose a nonempty setup name other than show or remove.")
    profile = load_config().profiles.get(setup.action_profile)
    checked_provider = None
    if profile is not None and profile.kind == "decision":
        try:
            asyncio.run(resolve(profile.provider))
        except CredentialMissing as exc:
            raise click.UsageError(str(exc)) from None
        checked_provider = profile.provider

    def edit(config: MobileConfig) -> None:
        action = config.profiles.get(setup.action_profile)
        if action is not None and action.kind == "decision" and action.provider != checked_provider:
            raise click.UsageError(f"Profile {setup.action_profile!r} changed; try again.")
        config.setups[name] = setup
        if make_default:
            config.default_setup = name
        elif clear_default:
            config.default_setup = None

    update_config(edit)
    click.echo(f"Setup saved: {name}")
    if make_default:
        click.echo(f"Default setup: {name}")
    elif clear_default:
        _echo_default_cleared()


def _echo_default_cleared() -> None:
    click.echo("Default setup cleared; agent run uses a straight agent with the first LLM profile.")


def _profile_choices(config: MobileConfig) -> tuple[list[SelectChoice], list[SelectChoice]]:
    choices, llms = [], []
    for name, profile in config.profiles.items():
        match profile:
            case LLMProfile(provider=provider, model=model):
                choice = SelectChoice(name, name, f"llm · {provider}/{model}")
                llms.append(choice)
            case DecisionProfile(provider=provider, vision=vision, min_confidence=threshold):
                confidence = f" · min {threshold:g}" if threshold is not None else ""
                choice = SelectChoice(
                    name, name, f"decision · {provider} · vision {vision or 'on'}{confidence}"
                )
        choices.append(choice)
    return choices, llms


def _interactive(name: str | None) -> None:
    if not menu.is_interactive():
        raise click.UsageError(
            "Supply a setup name and role flags, or open configure in a terminal."
        )
    config = load_config()
    choices, llms = _profile_choices(config)
    if not choices:
        raise click.UsageError("No profiles. Run: mobilerun agent profiles add --help")
    try:
        planning = (
            pick("Setup structure", ["Straight", "Manager + executor"]) == "Manager + executor"
        )
        if planning and not llms:
            raise click.UsageError("Add an LLM profile before configuring a planner.")
        planner = pick_choice("Planner profile", llms) if planning else None
        action = pick_choice("Executor profile" if planning else "Selector profile", choices)
        delegate = (
            pick_choice("Delegate (optional)", [SelectChoice("", "None"), *llms]) or None
            if config.profiles[action].kind == "decision"
            else None
        )
        if name is None:
            name = prompts.text_prompt(
                "Setup name", default=config.default_setup or "default"
            ).strip()
            if not name:
                raise click.Abort()
        default_choice = pick("Make default?", ["No", "Yes", "None (no default)"])
        if pick("Save setup?", ["Done", "Cancel"]) != "Done":
            raise click.Abort()
    except (click.Abort, KeyboardInterrupt, EOFError):
        click.echo("Cancelled.")
        return
    _save_setup(
        name,
        SetupConfig(
            selector=None if planning else action,
            planner=planner,
            executor=action if planning else None,
            delegate=delegate,
        ),
        default_choice == "Yes",
        default_choice == "None (no default)",
    )


@configure.command()
@click.argument("name", required=False)
def show(name: str | None) -> None:
    """Show a named setup or all setups and the default."""
    config = load_config()
    if name is None and config.setups and menu.is_interactive():
        try:
            name = (
                pick_choice(
                    "Setup to show",
                    [
                        SelectChoice("", "All setups"),
                        *(SelectChoice(key, key) for key in config.setups),
                    ],
                )
                or None
            )
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo("Cancelled.")
            return
    if name is not None and name not in config.setups:
        raise click.UsageError(f"Unknown setup {name!r}")
    setups = config.setups if name is None else {name: config.setups[name]}
    output = {
        "setups": {key: setup.model_dump(exclude_defaults=True) for key, setup in setups.items()}
    }
    click.echo(yaml.safe_dump(output, sort_keys=False).rstrip())  # type: ignore[no-untyped-call]
    if name is None:
        click.echo(f"Default setup: {config.default_setup or '(none)'}")


@configure.command()
@click.argument("name", required=False)
@click.option("--unset", "unset_default", is_flag=True, help="Clear the default setup.")
def default(name: str | None, unset_default: bool) -> None:
    """Use NAME when run does not specify --setup, or clear it with --unset."""
    if unset_default and name is not None:
        raise click.UsageError("NAME cannot be used with --unset.")
    if not unset_default and name is None:
        if not menu.is_interactive():
            raise click.UsageError("Supply a setup name, --unset, or open configure in a terminal.")
        choices = [SelectChoice("", "None (no default)")]
        choices.extend(SelectChoice(key, key) for key in load_config().setups)
        name = pick_choice("Default setup", choices) or None

    def edit(config: MobileConfig) -> None:
        if name is not None and name not in config.setups:
            raise click.UsageError(f"Unknown setup {name!r}")
        config.default_setup = name

    update_config(edit)
    if name is None:
        _echo_default_cleared()
    else:
        click.echo(f"Default setup: {name}")


@configure.command()
@click.argument("name")
def remove(name: str) -> None:
    """Remove a setup without removing its profiles."""
    was_default = False

    def edit(config: MobileConfig) -> None:
        nonlocal was_default
        if name not in config.setups:
            raise click.UsageError(f"Unknown setup {name!r}")
        was_default = config.default_setup == name
        if was_default:
            config.default_setup = None
        del config.setups[name]

    update_config(edit)
    click.echo(f"Setup removed: {name}")
    if was_default:
        _echo_default_cleared()

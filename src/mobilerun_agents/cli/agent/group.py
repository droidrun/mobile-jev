"""Agent commands."""

import click
from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.base.prompts import SelectChoice

_COMMANDS = {
    "configure": "Create, inspect and choose named setups.",
    "profiles": "Manage named model profiles.",
    "providers": "Manage provider credentials.",
    "run": "Run a command on your mobile device.",
    "task": "Inspect and stop cloud agent tasks.",
}


class AgentGroup(MenuGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_COMMANDS)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        with formatter.section("Commands"):
            formatter.write_dl(list(_COMMANDS.items()))

    def menu_choices(self, ctx: click.Context) -> list[SelectChoice]:
        return [SelectChoice(name, name, hint) for name, hint in _COMMANDS.items()]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name == "run":
            from .run import run

            return run
        if cmd_name == "task":
            from .task import task

            return task
        if cmd_name == "configure":
            from .configure import configure

            return configure
        if cmd_name == "profiles":
            from .profiles import profiles

            return profiles
        if cmd_name == "providers":
            from .providers import providers

            return providers
        return None


@click.group(cls=AgentGroup)
def agent() -> None:
    """Run an agent on a mobile device."""

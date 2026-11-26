"""Agent commands."""

import click
from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.base.prompts import SelectChoice

_COMMANDS = {
    "logout": "Remove a provider and its saved key.",
    "run": "Run a command on your mobile device.",
    "setup": "Manage providers, models and the default LLM.",
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
        if cmd_name in ("setup", "logout"):
            from .setup import logout, setup

            return setup if cmd_name == "setup" else logout
        return None


@click.group(cls=AgentGroup)
def agent() -> None:
    """Run an agent on a mobile device."""

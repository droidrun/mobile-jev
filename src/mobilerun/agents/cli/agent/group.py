"""Agent commands."""

from importlib import import_module

import click
from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.base.prompts import SelectChoice

_COMMANDS = {
    "run": ("mobilerun.agents.cli.agent.run", "Run a command on your mobile device."),
    "llm": ("mobilerun.agents.cli.agent.llm", "Configure LLM providers."),
    "replay": ("mobilerun.agents.cli.agent.replay", "Replay an agent macro."),
    "task": ("mobilerun.agents.cli.agent.task", "Inspect and stop cloud agent tasks."),
}


class AgentGroup(MenuGroup):
    def list_commands(self, ctx):
        return sorted(_COMMANDS)

    def format_commands(self, ctx, formatter):
        with formatter.section("Commands"):
            formatter.write_dl([(name, _COMMANDS[name][1]) for name in self.list_commands(ctx)])

    def menu_choices(self, ctx):
        return [SelectChoice(name, name, _COMMANDS[name][1]) for name in self.list_commands(ctx)]

    def get_command(self, ctx, cmd_name):
        if cmd_name not in _COMMANDS:
            return None
        return getattr(import_module(_COMMANDS[cmd_name][0]), cmd_name)


@click.group(cls=AgentGroup)
def agent() -> None:
    """Run an agent on a mobile device."""

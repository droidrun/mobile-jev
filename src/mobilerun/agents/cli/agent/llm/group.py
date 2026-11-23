import click
from mobilerun.cli.base.menu import MenuGroup

from mobilerun.agents.cli.agent.llm.configure import configure
from mobilerun.agents.cli.agent.llm.login import login


@click.group(cls=MenuGroup)
def llm() -> None:
    """Configure LLM providers."""


llm.add_command(configure)
llm.add_command(login)

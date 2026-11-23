import click
from mobilerun.cli.base.menu import MenuGroup

from mobilerun.agents.cli.agent.llm.login.anthropic import configure_anthropic
from mobilerun.agents.cli.agent.llm.login.gemini import configure_gemini
from mobilerun.agents.cli.agent.llm.login.openai import configure_openai
from mobilerun.agents.cli.agent.llm.login.xai import configure_xai


@click.group(cls=MenuGroup)
def login() -> None:
    """Log in to an LLM provider."""


for command in (configure_openai, configure_anthropic, configure_gemini, configure_xai):
    login.add_command(command)

import click

from .replay import replay


@click.group(name="macro-agent", commands={"replay": replay})
def macro_agent() -> None:
    """Replay a macro with agent recovery."""

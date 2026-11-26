"""CLI routing for add-on logging."""

import logging
import os
from contextlib import ExitStack

import click
from mobilerun.logs.handlers import CLILogHandler

from .diagnostics import private_values


class _AgentHandler(CLILogHandler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.__dict__.get("agent_result"):
            # Keep just the final answer on stdout for shell callers.
            click.echo(self.format(record))
        else:
            super().emit(record)


def _shared_log(record: logging.LogRecord) -> bool:
    # Base Portal and some add-on selectors both log directly to 'mobilerun'.
    return record.levelno >= logging.INFO or "/mobilerun_agents/" in record.pathname.replace(
        "\\", "/"
    )


def configure_cli_logging(debug: bool) -> None:
    """Enable agent progress, independently of base transport verbosity."""
    restore = ExitStack()
    token = private_values.set(set() if debug else None)
    restore.callback(private_values.reset, token)
    context = click.get_current_context(silent=True)
    if context:
        context.call_on_close(restore.close)

    def install(logger: logging.Logger, handler: logging.Handler) -> None:
        logger.addHandler(handler)
        restore.callback(handler.close)
        restore.callback(logger.removeHandler, handler)

    transport_debug = os.environ.get("MOBILERUN_LOG_LEVEL", "").lower() == "debug"
    agent = logging.getLogger("mobilerun_agents")
    restore.callback(agent.setLevel, agent.level)
    restore.callback(setattr, agent, "propagate", agent.propagate)
    agent.setLevel(logging.DEBUG if debug or transport_debug else logging.INFO)
    if not any(type(handler) is _AgentHandler for handler in agent.handlers):
        install(agent, _AgentHandler())
    agent.propagate = False

    shared = logging.getLogger("mobilerun")
    restore.callback(shared.setLevel, shared.level)
    shared.setLevel(agent.level)
    if transport_debug and _shared_log in shared.filters:
        restore.callback(shared.addFilter, _shared_log)
        shared.removeFilter(_shared_log)
    elif not transport_debug and _shared_log not in shared.filters:
        restore.callback(shared.removeFilter, _shared_log)
        shared.addFilter(_shared_log)
    for name in ("mobilerun.clients.portal", "httpx", "httpcore", "urllib3"):
        logger = logging.getLogger(name)
        restore.callback(logger.setLevel, logger.level)
        logger.setLevel(logging.DEBUG if transport_debug else logging.INFO)
        if transport_debug and not logger.hasHandlers():
            install(logger, CLILogHandler())

import logging

import click
import pytest

from mobilerun_agents.agent.logs import configure_cli_logging


@pytest.mark.parametrize(
    "debug,env", [(False, None), (True, None), (True, "DEBUG"), (False, "debug")]
)
def test_cli_logging_levels_and_routing(debug, env, monkeypatch, capsys, caplog):
    if env:
        monkeypatch.setenv("MOBILERUN_LOG_LEVEL", env)
    else:
        monkeypatch.delenv("MOBILERUN_LOG_LEVEL", raising=False)
    transport = [
        logging.getLogger(name)
        for name in ("mobilerun", "mobilerun.clients.portal", "httpx", "httpcore", "urllib3")
    ]
    agent = logging.getLogger("mobilerun_agents")
    for logger in [*transport, agent]:
        monkeypatch.setattr(logger, "level", logging.NOTSET)
        monkeypatch.setattr(logger, "propagate", True)
        monkeypatch.setattr(logger, "handlers", [])
        monkeypatch.setattr(logger, "filters", [])
    configure_cli_logging(debug)
    configure_cli_logging(debug)
    for logger in transport:
        logger.debug("transport diagnostic")
    logging.getLogger("mobilerun_agents.agent.selector").debug("agent debug")
    logging.getLogger("mobilerun_agents.agent.logs.console").info("progress")
    logging.getLogger("mobilerun_agents.agent.logs.console").info(
        "answer", extra={"agent_result": True}
    )
    streams = capsys.readouterr()
    assert streams.out == "answer\n"
    assert streams.err.count("progress") == 1
    assert ("agent debug" in streams.err) == (debug or bool(env))
    assert caplog.messages.count("transport diagnostic") == (len(transport) if env else 0)


def test_cli_context_restores_sdk_logging():
    agent = logging.getLogger("mobilerun_agents")
    shared = logging.getLogger("mobilerun")
    previous = agent.level, agent.propagate, agent.handlers.copy(), shared.filters.copy()
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
    assert (agent.level, agent.propagate, agent.handlers, shared.filters) == previous

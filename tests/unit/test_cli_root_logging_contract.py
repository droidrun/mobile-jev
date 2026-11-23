import importlib
import logging

from click.testing import CliRunner


def test_root_reuses_its_installed_handler(monkeypatch):
    app = importlib.import_module("mobilerun.cli.app")
    state = importlib.import_module("mobilerun.cli.base.logging")
    logger = logging.getLogger("mobilerun")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    monkeypatch.setattr(state, "_handler", None)
    runner = CliRunner()
    runner.invoke(app.cli, ["device", "--help"])
    handler = state._handler
    runner.invoke(app.cli, ["device", "--help"])
    assert state._handler is handler
    assert logger.handlers.count(handler) == 1


def test_root_reuses_handler_installed_by_agent_cli(monkeypatch):
    from mobilerun.agents.cli.agent.run import _show_local_config
    from mobilerun.agents.config.user_config import MobileConfig

    app = importlib.import_module("mobilerun.cli.app")
    state = importlib.import_module("mobilerun.cli.base.logging")
    logger = logging.getLogger("mobilerun")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
    monkeypatch.setattr(state, "_handler", None)
    _show_local_config(MobileConfig(), False, "quiet")
    handler = state._handler
    CliRunner().invoke(app.cli, ["device", "--help"])
    assert logger.handlers == [handler]

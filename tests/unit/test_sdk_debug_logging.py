import json
import logging

import pytest
from mobilerun.clients.portal.setup import portal_version_kwargs

import mobilerun
from mobilerun.agents.workflows.droid import droid_agent as agent_module


@pytest.fixture
def mobilerun_logger():
    log = logging.getLogger("mobilerun")
    saved = (list(log.handlers), log.level, log.propagate)
    yield log
    log.handlers, log.level, log.propagate = saved[0], saved[1], saved[2]


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_agent_does_not_configure_logging_and_propagates_to_root(
    mobilerun_logger, monkeypatch, caplog
):
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agents.config.user_config import MobileConfig, TelemetryConfig

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    mobilerun_logger.handlers = []
    mobilerun_logger.propagate = True
    mobilerun_logger.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG):
        agent_module.MobileAgent(
            "Open the app",
            config=MobileConfig(telemetry=TelemetryConfig(enabled=False)),
            llms=MockLLM(),
        )
    assert mobilerun_logger.handlers == []
    assert mobilerun_logger.propagate is True
    assert "MobileAgent initialized successfully." in caplog.messages


def test_sdk_debug_config_logs_app_card_loading(mobilerun_logger, monkeypatch, tmp_path):
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agents.config.user_config import (
        AgentConfig,
        AppCardConfig,
        LoggingConfig,
        MobileConfig,
        TelemetryConfig,
    )

    (tmp_path / "android").mkdir()
    (tmp_path / "android" / "app_cards.json").write_text(json.dumps({"com.example": "a.md"}))
    mobilerun_logger.setLevel(logging.DEBUG)
    records = _Records()
    mobilerun_logger.addHandler(records)
    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)

    agent_module.MobileAgent(
        "Open the app",
        config=MobileConfig(
            agent=AgentConfig(reasoning=True, app_cards=AppCardConfig(app_cards_dir=str(tmp_path))),
            logging=LoggingConfig(debug=True),
            telemetry=TelemetryConfig(enabled=False),
        ),
        llms=MockLLM(),
    )

    assert "Loaded android/app_cards.json with 1 entries" in records.messages


def test_portal_setup_gets_mobilerun_version_when_supported():
    def new_setup(device, debug=False, version=None):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup) == {}


def test_portal_setup_can_ask_for_an_in_place_update():
    def new_setup(device, debug=False, *, version=None, uninstall=True):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup, in_place=True) == {
        "version": mobilerun.__version__,
        "uninstall": False,
    }
    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup, in_place=True) == {}


def test_sdk_version_check_still_flags_an_rc_after_its_release(monkeypatch):
    import asyncio

    from mobilerun.diagnostics import checks as doctor
    from mobilerun.diagnostics import versions as sdk_releases

    class Response:
        status_code = 200

        def json(self):
            return {"tag_name": "v0.6.22"}

    monkeypatch.setattr(doctor, "__version__", "0.6.22rc1")
    monkeypatch.setattr(sdk_releases.requests, "get", lambda url, **kwargs: Response())

    result = asyncio.run(doctor.check_sdk_version(debug=False))

    assert result.status == doctor.Status.WARN

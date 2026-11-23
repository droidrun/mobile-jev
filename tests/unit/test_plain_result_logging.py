import logging
from io import StringIO

import pytest
from mobilerun.logs.handlers import CLILogHandler
from rich.console import Console

from mobilerun.agents.logs.events import EventHandler
from mobilerun.agents.workflows.droid.events import FinalizeEvent


@pytest.fixture(autouse=True)
def isolate_mobilerun_logger():
    logger = logging.getLogger("mobilerun")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.mark.parametrize(
    ("success", "expected", "symbol"),
    [
        (True, "Done in 0 steps", "✓"),
        (False, "Failed after 0 steps: failed", "✗"),
    ],
)
def test_library_record_is_plain_but_cli_has_status_symbol(caplog, success, expected, symbol):
    with caplog.at_level(logging.WARNING, logger="mobilerun"):
        EventHandler().handle(FinalizeEvent(success=success, reason="failed"))
    record = caplog.records[-1]
    assert record.getMessage() == expected
    stream = StringIO()
    handler = CLILogHandler()
    handler.console = Console(file=stream, force_terminal=False, width=120)
    handler.emit(record)
    assert stream.getvalue() == f"{symbol} {expected}\n"


def test_cli_uses_plain_result_when_terminal_is_ascii(caplog):
    class AsciiStream(StringIO):
        @property
        def encoding(self):
            return "ascii"

    with caplog.at_level(logging.WARNING, logger="mobilerun"):
        EventHandler().handle(FinalizeEvent(success=True, reason="finished"))
    stream = AsciiStream()
    handler = CLILogHandler()
    handler.console = Console(file=stream, force_terminal=False, width=120)
    handler.emit(caplog.records[-1])
    assert stream.getvalue() == "Done in 0 steps\n"

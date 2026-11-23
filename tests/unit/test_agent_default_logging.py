"""SDK logging is opt-in through agent execution, not package import."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("entry", ["mobile", "operator"])
@pytest.mark.parametrize("configured", ["none", "root", "mobilerun"])
def test_agent_entry_installs_default_only_without_application_logging(entry, configured):
    script = f"""
import asyncio
import logging
from unittest.mock import patch

import mobilerun
assert not logging.getLogger("mobilerun").handlers

from mobilerun.agents.workflows.droid.droid_agent import MobileAgent, Workflow
from mobilerun.agents.operators.agent import AgentOperator
from mobilerun.logs.handlers import CLILogHandler, configure_logging

logger = logging.getLogger("mobilerun")
root = logging.getLogger()
assert not logger.handlers
configured = {configured!r}
handler = logging.NullHandler()
if configured != "none":
    target = root if configured == "root" else logger
    target.addHandler(handler)
    logger.setLevel(logging.ERROR)
before = logger.handlers[:], logger.level, logger.propagate

def invoke():
    if {entry!r} == "mobile":
        agent = MobileAgent.__new__(MobileAgent)
        agent._tracing_session_id = None
        agent._tracing_user_id = None
        with patch.object(Workflow, "run", return_value="started"):
            assert agent.run() == "started"
    else:
        with patch("mobilerun.agents.operators.agent.ConfigLoader.load", side_effect=RuntimeError("stop before LLM")):
            try:
                asyncio.run(AgentOperator().execute(None, "test"))
            except RuntimeError as error:
                assert str(error) == "stop before LLM"

invoke()
if configured == "none":
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], CLILogHandler)
    assert logger.level == logging.INFO and not logger.propagate
else:
    assert (logger.handlers, logger.level, logger.propagate) == before
first = logger.handlers[:]
invoke()
assert logger.handlers == first
replacement = logging.NullHandler()
configure_logging(False, replacement)
invoke()
assert logger.handlers == [replacement]
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""Compatibility alias for mobilerun.agents.errors.agent_timeout."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.errors.agent_timeout")

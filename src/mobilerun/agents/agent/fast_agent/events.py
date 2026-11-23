"""Compatibility alias for mobilerun.agents.workflows.fast_agent.events."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.workflows.fast_agent.events")

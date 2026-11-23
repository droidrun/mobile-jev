"""Compatibility alias for mobilerun.agents.workflows.fast_agent.native_tools."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.workflows.fast_agent.native_tools")

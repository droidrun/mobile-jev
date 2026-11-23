"""Compatibility alias for mobilerun.agents.tools.actions."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.tools.actions")

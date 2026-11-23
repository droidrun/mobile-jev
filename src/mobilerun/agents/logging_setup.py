"""Compatibility alias for mobilerun.agents.logs.setup."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.logs.setup")

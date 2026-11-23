"""Compatibility alias for mobilerun.agents.tools.signatures."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.tools.signatures")

"""Compatibility alias for mobilerun.agents.workflows.droid.state."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.workflows.droid.state")

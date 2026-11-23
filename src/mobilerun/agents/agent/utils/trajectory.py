"""Compatibility alias for mobilerun.agents.trajectory.trajectory."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.trajectory.trajectory")

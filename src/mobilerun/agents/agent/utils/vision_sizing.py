"""Compatibility alias for mobilerun.agents.utils.vision_sizing."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.utils.vision_sizing")

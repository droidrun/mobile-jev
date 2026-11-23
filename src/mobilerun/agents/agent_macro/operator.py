"""Compatibility alias for mobilerun.agents.operators.agent_macro."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.operators.agent_macro")

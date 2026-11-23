"""Compatibility alias for mobilerun.agents.workflows.manager.prompts."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.workflows.manager.prompts")

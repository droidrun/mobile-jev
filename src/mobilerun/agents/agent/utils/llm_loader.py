"""Compatibility alias for mobilerun.agents.llm.llm_loader."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.llm.llm_loader")

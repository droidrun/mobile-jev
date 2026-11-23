"""Compatibility alias for mobilerun.agents.llm.providers.grok."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.llm.providers.grok")

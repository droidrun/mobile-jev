"""Compatibility alias for mobilerun.agents.llm.oauth.anthropic_oauth_llm."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.llm.oauth.anthropic_oauth_llm")

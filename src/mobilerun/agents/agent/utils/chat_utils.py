"""Compatibility alias for mobilerun.agents.utils.chat_utils."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mobilerun.agents.utils.chat_utils")

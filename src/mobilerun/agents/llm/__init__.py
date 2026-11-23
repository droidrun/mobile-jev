"""Lazy access to agent LLM loading functions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agents.llm.llm_loader import load_agent_llms
    from mobilerun.agents.llm.llm_picker import load_llm

_EXPORTS = {
    "load_llm": "mobilerun.agents.llm.llm_picker",
    "load_agent_llms": "mobilerun.agents.llm.llm_loader",
}
__all__ = ["load_llm", "load_agent_llms"]


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

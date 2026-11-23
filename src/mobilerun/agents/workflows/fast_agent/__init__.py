"""Public exports are loaded on demand."""

from importlib import import_module

__all__ = ["FastAgent"]

_EXPORTS = {"FastAgent": "mobilerun.agents.workflows.fast_agent.fast_agent"}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)

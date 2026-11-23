"""Public exports are loaded on demand."""

from importlib import import_module

__all__ = ["ResultEvent"]

_EXPORTS = {
    "ResultEvent": "mobilerun.agents.workflows.droid.events",
    "WorkflowOptions": "mobilerun.agents.workflows.options",
    "WorkflowOptionsWithoutTimeout": "mobilerun.agents.workflows.options",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)

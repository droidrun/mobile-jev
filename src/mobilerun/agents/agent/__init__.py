"""Lazy compatibility exports for the former agent package."""

from importlib import import_module

_EXPORTS = {
    "ResultEvent": "mobilerun.agents.workflows.droid.events",
    "WorkflowOptions": "mobilerun.agents.workflows.options",
    "WorkflowOptionsWithoutTimeout": "mobilerun.agents.workflows.options",
}
__all__ = ["ResultEvent"]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(name)

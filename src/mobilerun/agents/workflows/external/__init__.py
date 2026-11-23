"""Public exports are loaded on demand."""

from importlib import import_module

__all__ = ["ExternalAgentModule", "list_agents", "load_agent", "run_with_context"]

_EXPORTS = {
    "ExternalAgentModule": "mobilerun.agents.workflows.external.loader",
    "list_agents": "mobilerun.agents.workflows.external.loader",
    "load_agent": "mobilerun.agents.workflows.external.loader",
    "run_with_context": "mobilerun.agents.workflows.external.loader",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)

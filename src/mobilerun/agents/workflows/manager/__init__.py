"""Public exports are loaded on demand."""

from importlib import import_module

__all__ = [
    "ManagerAgent",
    "StatelessManagerAgent",
    "ManagerInputEvent",
    "ManagerPlanEvent",
    "ManagerContextEvent",
    "ManagerResponseEvent",
    "ManagerPlanDetailsEvent",
    "parse_manager_response",
]

_EXPORTS = {
    "ManagerAgent": "mobilerun.agents.workflows.manager.manager_agent",
    "StatelessManagerAgent": "mobilerun.agents.workflows.manager.stateless_manager_agent",
    "ManagerInputEvent": "mobilerun.agents.workflows.droid.events",
    "ManagerPlanEvent": "mobilerun.agents.workflows.droid.events",
    "ManagerContextEvent": "mobilerun.agents.workflows.manager.events",
    "ManagerResponseEvent": "mobilerun.agents.workflows.manager.events",
    "ManagerPlanDetailsEvent": "mobilerun.agents.workflows.manager.events",
    "parse_manager_response": "mobilerun.agents.workflows.manager.prompts",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)

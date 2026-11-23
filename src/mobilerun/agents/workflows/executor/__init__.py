"""Public exports are loaded on demand."""

from importlib import import_module

__all__ = [
    "ExecutorAgent",
    "ExecutorInputEvent",
    "ExecutorResultEvent",
    "ExecutorContextEvent",
    "ExecutorResponseEvent",
    "ExecutorActionEvent",
    "ExecutorActionResultEvent",
]

_EXPORTS = {
    "ExecutorAgent": "mobilerun.agents.workflows.executor.executor_agent",
    "ExecutorInputEvent": "mobilerun.agents.workflows.droid.events",
    "ExecutorResultEvent": "mobilerun.agents.workflows.droid.events",
    "ExecutorContextEvent": "mobilerun.agents.workflows.executor.events",
    "ExecutorResponseEvent": "mobilerun.agents.workflows.executor.events",
    "ExecutorActionEvent": "mobilerun.agents.workflows.executor.events",
    "ExecutorActionResultEvent": "mobilerun.agents.workflows.executor.events",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name]), name)

"""Compatibility package; implementations live in mobilerun.agents.workflows.droid."""

from importlib import import_module


def __getattr__(name: str) -> object:
    return getattr(import_module("mobilerun.agents.workflows.droid"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module("mobilerun.agents.workflows.droid"))))

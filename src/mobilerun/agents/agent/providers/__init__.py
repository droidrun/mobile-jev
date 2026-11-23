"""Compatibility package; implementations live in mobilerun.agents.llm.providers."""

from importlib import import_module


def __getattr__(name: str) -> object:
    return getattr(import_module("mobilerun.agents.llm.providers"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(import_module("mobilerun.agents.llm.providers"))))

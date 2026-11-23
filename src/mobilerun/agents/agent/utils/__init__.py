"""Lazy compatibility exports for the former agent utility package."""

from importlib import import_module

_EXPORTS = {
    "to_chat_messages": "mobilerun.agents.utils.chat_utils",
    "has_content": "mobilerun.agents.utils.chat_utils",
    "filter_empty_messages": "mobilerun.agents.utils.chat_utils",
    "limit_history": "mobilerun.agents.utils.chat_utils",
    "PromptResolver": "mobilerun.agents.utils.prompt_resolver",
    "build_tool_registry": "mobilerun.agents.tools.signatures",
    "Trajectory": "mobilerun.agents.trajectory.trajectory",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(name)

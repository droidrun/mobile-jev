"""Agent utilities, loaded on demand to avoid importing optional providers."""

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

__all__ = [
    "to_chat_messages",
    "has_content",
    "filter_empty_messages",
    "limit_history",
    "PromptResolver",
    "build_tool_registry",
    "Trajectory",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value

"""Lazy exports for the agent operators."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agents.operators.agent import AgentOperator
    from mobilerun.agents.operators.agent_macro import AgentMacroOperator
    from mobilerun.agents.operators.delegation import DelegationOperator

__all__ = ["AgentOperator", "AgentMacroOperator", "DelegationOperator"]

_EXPORTS = {
    "AgentOperator": "mobilerun.agents.operators.agent",
    "AgentMacroOperator": "mobilerun.agents.operators.agent_macro",
    "DelegationOperator": "mobilerun.agents.operators.delegation",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value

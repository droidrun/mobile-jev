"""Public API for composing and running a mobile agent."""

from typing import TYPE_CHECKING

from .agent.action_selector import (
    ActionSelector,
    DecisionSelector,
    DelegatingSelector,
    FallbackSelector,
    LlamaIndexLLMSelector,
    LLMSelector,
    Vision,
)
from .agent.operators import AgentOperator, ExternalOperator, PlanningOperator
from .agent.planner import LlamaIndexPlanner, Planner
from .agent.run import Run
from .agent.tools import Tool, ToolRegistry, tool
from .agent.tools.base import Source
from .agent.workflows.agent import AgentWorkflow
from .agent.workflows.core.result import AgentResult
from .config.user_config import MobileConfig

if TYPE_CHECKING:
    from .macro_agent.operator import MacroAgentOperator


def __getattr__(name: str) -> object:
    if name == "MacroAgentOperator":
        from .macro_agent.operator import MacroAgentOperator

        return MacroAgentOperator
    raise AttributeError(name)


__all__ = [
    "ActionSelector",
    "AgentOperator",
    "ExternalOperator",
    "PlanningOperator",
    "Planner",
    "LlamaIndexPlanner",
    "AgentResult",
    "ToolRegistry",
    "Run",
    "Source",
    "Tool",
    "tool",
    "AgentWorkflow",
    "DecisionSelector",
    "DelegatingSelector",
    "FallbackSelector",
    "LLMSelector",
    "LlamaIndexLLMSelector",
    "MacroAgentOperator",
    "MobileConfig",
    "Vision",
]

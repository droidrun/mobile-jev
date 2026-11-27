"""Planner contract and decisions applied by the planning workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.memory import Memory
from llama_index.core.tools import BaseTool, ToolSelection
from mobilerun.models.context import JSONValue
from workflows import Context

from mobilerun_agents.agent.workflows.core.state import PlanningState


def remaining_steps(state: PlanningState) -> int:
    return state.max_steps - state.planner_turns - state.executor_steps


def remaining_child_steps(state: PlanningState) -> int:
    return remaining_steps(state) - 1


@dataclass(frozen=True)
class Plan:
    subgoals: tuple[str, ...]
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finish:
    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


class Planner(ABC):
    def with_tools(self, tools: tuple[BaseTool, ...], feedback: str | None) -> "Planner":
        """Bind optional memory tools and the previous tool result for one turn."""
        return self

    def bind(
        self,
        output_schema: dict[str, JSONValue] | None,
        repair: str | None,
    ) -> "Planner":
        """Bind optional prompt inputs for a turn; custom planners can just read the context."""
        return self

    @abstractmethod
    async def plan(
        self, ctx: Context[PlanningState], memory: Memory
    ) -> Plan | Finish | ToolSelection:
        """Return a plan, finish, or memory tool call without modifying state."""

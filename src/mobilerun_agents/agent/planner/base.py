"""Planner contract and decisions applied by the planning workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mobilerun.models.context import JSONValue
from workflows import Context

from mobilerun_agents.agent.workflows.core.state import PlanningState


@dataclass(frozen=True)
class Plan:
    subgoals: tuple[str, ...]


@dataclass(frozen=True)
class Finish:
    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


class Planner(ABC):
    def bind(
        self, output_schema: dict[str, JSONValue] | None, repair: str | None, observe_screen: bool
    ) -> "Planner":
        """Bind optional prompt inputs for a turn; custom planners can just read the context."""
        return self

    @abstractmethod
    async def plan(self, ctx: Context[PlanningState]) -> Plan | Finish:
        """Return subgoals or a final result without modifying workflow state."""

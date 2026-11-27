"""Planner contract and decisions applied by the planning workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llama_index.core.memory import Memory
from mobilerun.models.context import JSONValue
from workflows import Context

from mobilerun_agents.agent.logs.redaction import Redactor
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
    def bind(
        self,
        output_schema: dict[str, JSONValue] | None,
        repair: str | None,
        redact: Redactor,
    ) -> "Planner":
        """Bind optional prompt inputs for a turn; custom planners can just read the context."""
        return self

    @abstractmethod
    async def plan(self, ctx: Context[PlanningState], memory: Memory) -> Plan | Finish:
        """Return subgoals and optional findings or a final result without modifying state."""

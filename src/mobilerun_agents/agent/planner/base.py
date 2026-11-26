"""Planner contract and decisions applied by the planning workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mobilerun.models.context import JSONValue
from pydantic import BaseModel

from mobilerun_agents.config.user_config import AgentConfig


class SubgoalResult(BaseModel):
    goal: str
    success: bool
    message: str
    steps: int


@dataclass(frozen=True)
class PlanningInput:
    goal: str
    config: AgentConfig
    remaining_plan: tuple[str, ...] = ()
    results: tuple[SubgoalResult, ...] = ()
    notes: tuple[str, ...] = ()
    package: str | None = None
    activity: str | None = None
    ui: str = ""
    output: type[BaseModel] | None = None
    repair: str | None = None


@dataclass(frozen=True)
class Plan:
    subgoals: tuple[str, ...]


@dataclass(frozen=True)
class Finish:
    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


class Planner(ABC):
    @abstractmethod
    async def plan(self, request: PlanningInput) -> Plan | Finish:
        """Return subgoals or a final result without modifying workflow state."""

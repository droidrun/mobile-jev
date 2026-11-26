"""Planner contract and decisions applied by the planning workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mobilerun.models.context import JSONValue
from pydantic import BaseModel

from mobilerun_agents.agent.workflows.core.result import FailureSummary
from mobilerun_agents.config.user_config import AgentConfig


class SubgoalResult(BaseModel):
    goal: str
    success: bool
    message: str
    steps: int
    failure: FailureSummary | None = None


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
    observe_screen: bool = False
    screenshot: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Plan:
    subgoals: tuple[str, ...]


@dataclass(frozen=True)
class Finish:
    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


class Planner(ABC):
    @property
    def supports_images(self) -> bool:
        return False

    @abstractmethod
    async def plan(self, request: PlanningInput) -> Plan | Finish:
        """Return subgoals or a final result without modifying workflow state."""

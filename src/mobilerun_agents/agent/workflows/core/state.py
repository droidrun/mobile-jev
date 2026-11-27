from typing import Literal

from llama_index.core.tools import ToolSelection
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, Field

from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS

from .snapshot import StepSnapshot


class JevAbstention(BaseModel):
    choice: str
    confidence: float | None
    target_confidence: float | None
    cutoff: float
    thought: str | None
    choices: str


class Outcome(BaseModel):
    success: bool
    message: str | None = None
    reason: str
    output: JSONValue = None
    jev_abstention: JevAbstention | None = None


class ActionRecord(BaseModel):
    step: int
    call: ToolSelection
    ok: bool
    summary: str
    thought: str | None = None
    before: StepSnapshot
    screen_changed: bool | None = None


class AgentState(BaseModel):
    goal: str = ""
    notes: list[str] = Field(default_factory=list)
    actions: list[ActionRecord] = Field(default_factory=list)
    current: StepSnapshot | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    repairs: int = 0
    repair: str | None = None
    retry_with_delegate: bool = False
    delegate_intent: str = ""
    outcome: Outcome | None = None


class FailureSummary(BaseModel):
    blockers: list[str]
    thoughts: list[str]
    screens: list[tuple[str, str, int]]
    actions: list[str]


class SubgoalResult(BaseModel):
    subgoal: str
    success: bool
    message: str | None = None
    reason: str
    summary: FailureSummary | None = None
    jev_abstention: JevAbstention | None = None
    before_screen: tuple[str | None, str | None] | None = None
    after_screen: tuple[str | None, str | None] | None = None
    screen_changed: bool | None = None


class PlanningState(BaseModel):
    goal: str = ""
    executor: Literal["decision", "decision_delegate", "llm", "custom"] = "custom"
    plan: list[str] = Field(default_factory=list)
    results: list[SubgoalResult] = Field(default_factory=list)
    planner_turns: int = 0
    executor_steps: int = 0
    actions: list[ActionRecord] = Field(default_factory=list)
    current: StepSnapshot | None = None
    notes: list[str] = Field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS

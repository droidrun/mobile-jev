from llama_index.core.tools import ToolSelection
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, Field

from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS

from .snapshot import StepSnapshot


class Outcome(BaseModel):
    success: bool
    message: str | None = None
    reason: str
    output: JSONValue = None


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
    retry_with_delegate: bool = False
    delegate_intent: str = ""
    outcome: Outcome | None = None


class SubgoalResult(BaseModel):
    subgoal: str
    success: bool
    message: str | None = None
    reason: str
    summary: str = ""


class PlanningState(BaseModel):
    goal: str = ""
    plan: list[str] = Field(default_factory=list)
    results: list[SubgoalResult] = Field(default_factory=list)
    planner_turns: int = 0
    executor_steps: int = 0
    actions: list[ActionRecord] = Field(default_factory=list)
    current: StepSnapshot | None = None
    notes: list[str] = Field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS

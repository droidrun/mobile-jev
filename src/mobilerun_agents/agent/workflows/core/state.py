from mobilerun.models.context import JSONValue
from pydantic import BaseModel, Field

from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS

from .result import ScreenVisit
from .snapshot import StepSnapshot


class ActionOutcome(BaseModel):
    step: int
    name: str
    args: dict[str, JSONValue]
    ok: bool
    replayable: bool = True
    summary: str
    thought: str | None = None
    screen_changed: bool | None = None


class ActionRecord(ActionOutcome):
    before: StepSnapshot = Field(repr=False)
    call_id: str = Field(default_factory=lambda: f"call_{uuid4().hex}", repr=False)


class ActionRepeat(BaseModel):
    name: str
    signature: str
    screen: str
    location: str
    count: int


class AgentState(BaseModel):
    max_steps: int = DEFAULT_MAX_STEPS
    notes: list[str] = Field(default_factory=list)
    goal: str = ""
    step: int = 0
    actions: list[ActionRecord] = Field(default_factory=list)
    repair_attempts: int = 0
    output_attempts: int = 0
    repeat: ActionRepeat | None = None
    pending_thoughts: list[str] = Field(default_factory=list, repr=False)
    no_progress: int = 0
    screens: dict[str, ScreenVisit] = Field(default_factory=dict)
    discoveries: dict[str, int] = Field(default_factory=dict)
    cycle_visits: dict[str, int] = Field(default_factory=dict)
    last_screen: str | None = None
    answer: str = ""
    reason: str = ""
from uuid import uuid4

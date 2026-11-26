from mobilerun.models.context import JSONValue
from pydantic import BaseModel, Field

from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS


class ActionRecord(BaseModel):
    step: int
    name: str
    args: dict[str, JSONValue]
    ok: bool
    replayable: bool = True
    summary: str
    screenshot: str | None
    ui_state: str | None


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

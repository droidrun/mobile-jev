"""Live workflow events, including safe tool selections and device-step links."""

from llama_index.core.tools import ToolSelection
from pydantic import Field, field_validator
from workflows.events import Event

from .workflows.core.privacy import mask_value
from .workflows.core.state import ActionRecord, Outcome, SubgoalResult


class Observed(Event):
    step: int
    package: str | None = None
    activity: str | None = None
    planner: bool = False


class Selected(Event):
    step: int
    calls: list[ToolSelection]
    thought: str | None = None
    confidence: float | None = None
    abstained: bool = False


class Acted(Event):
    action: ActionRecord
    step_ids: list[int] = Field(default_factory=list)


class Repaired(Event):
    step: int
    attempt: int
    reason: str


class Subgoal(Event):
    text: str
    result: SubgoalResult | None = None
    step: int | None = None
    budget: int | None = None


class Detail(Event):
    section: str
    text: str

    @field_validator("text")
    @classmethod
    def _redact_text(cls, text: str) -> str:
        return str(mask_value(text))


class Finished(Event):
    outcome: Outcome
    steps: int
    planner_turns: int | None = None
    budget: int | None = None

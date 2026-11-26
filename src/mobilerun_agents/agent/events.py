"""Raw workflow events. Redaction belongs to stream consumers."""

from llama_index.core.tools import ToolSelection
from workflows.events import Event

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


class Finished(Event):
    outcome: Outcome
    steps: int
    planner_turns: int | None = None
    budget: int | None = None

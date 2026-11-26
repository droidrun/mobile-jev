from mobilerun.models.context import ContextItem, JSONValue
from pydantic import BaseModel, Field

from mobilerun_agents.agent.workflows.core.state import ActionRecord


class StepSnapshot(BaseModel):
    notes: list[str] = Field(default_factory=list)
    goal: str
    step: int

    last_actions: list[ActionRecord]
    package: str | None
    activity: str | None
    ui_elements: list[dict[str, JSONValue]]
    device_context: list[ContextItem]

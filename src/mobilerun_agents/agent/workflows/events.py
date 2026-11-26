from typing import Any, Literal

from llama_index.core.workflow import Event
from pydantic import Field
from workflows.events import HumanResponseEvent

from mobilerun_agents.agent.action_selector.base import ActionSelection
from mobilerun_agents.agent.logs.diagnostics import DebugStep as DebugStep
from mobilerun_agents.agent.logs.diagnostics import Detail
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.state import ActionOutcome


class Observe(Event):
    pass


class Select(Event):
    pass


class Observed(Event):
    step: int
    package: str | None
    activity: str | None


class Act(Event):
    selection: ActionSelection


class Invalid(Event):
    error: str
    diagnostic: str = "Invalid tool arguments"


class GuardDetail(Detail):
    section: Literal["guard"] = "guard"  # type: ignore[assignment]


class Subgoal(Event):
    text: str


class StepFinished(Event):
    action: ActionOutcome
    # The first primitive carries the user-facing action; later ones can be
    # additional primitives emitted by the same tool call.
    display_action: ActionOutcome | None = Field(default=None, exclude=True, repr=False)
    delegated: bool = Field(default=False, exclude=True, repr=False)


class UserMessage(HumanResponseEvent):
    text: str


class Done(Event):
    result: AgentResult[Any]

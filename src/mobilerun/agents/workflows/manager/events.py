"""
Events for the ManagerAgent workflow.

Internal events for streaming to frontend/logging.
For MobileAgent coordination events, see droid/events.py
"""

from typing import TYPE_CHECKING, Optional

from mobilerun.agents.llm.usage import UsageResult
from mobilerun.agents.workflows.common.events import AgentEvent

if TYPE_CHECKING:
    from mobilerun.agents.logs.events import EventHandler


class ManagerContextEvent(AgentEvent):
    """Context prepared, ready for LLM call."""

    pass

    def log(self, out: "EventHandler") -> None:
        out.on_manager_context_event(self)


class ManagerResponseEvent(AgentEvent):
    """LLM response received, ready for parsing."""

    response: str
    usage: Optional[UsageResult] = None

    def log(self, out: "EventHandler") -> None:
        out.on_manager_response_event(self)


class ManagerPlanDetailsEvent(AgentEvent):
    """Plan parsed and ready (internal event with full details)."""

    plan: str
    subgoal: str
    thought: str
    answer: str = ""
    memory_update: str = ""
    progress_summary: str = ""
    success: Optional[bool] = None  # True/False if complete, None if in progress
    full_response: str = ""

    def log(self, out: "EventHandler") -> None:
        out.on_manager_plan_details_event(self)

"""
Events for the ExecutorAgent workflow.

Internal events for streaming to frontend/logging.
For MobileAgent coordination events, see droid/events.py
"""

from typing import TYPE_CHECKING, Dict, Optional

from mobilerun.agents.llm.usage import UsageResult
from mobilerun.agents.workflows.common.events import AgentEvent

if TYPE_CHECKING:
    from mobilerun.agents.logs.events import EventHandler


class ExecutorContextEvent(AgentEvent):
    """Context prepared, ready for LLM call."""

    subgoal: str


class ExecutorResponseEvent(AgentEvent):
    """LLM response received, ready for parsing."""

    response: str
    usage: Optional[UsageResult] = None


class ExecutorActionEvent(AgentEvent):
    """Action parsed, ready to execute."""

    action_json: str
    thought: str
    description: str
    full_response: str = ""

    def log(self, out: "EventHandler") -> None:
        out.on_executor_action_event(self)


class ExecutorActionResultEvent(AgentEvent):
    """Action execution result (internal event with full details)."""

    action: Dict
    success: bool
    error: str
    summary: str
    thought: str = ""
    full_response: str = ""

    def log(self, out: "EventHandler") -> None:
        out.on_executor_action_result_event(self)

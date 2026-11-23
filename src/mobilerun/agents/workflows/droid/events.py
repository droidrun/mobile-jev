"""
MobileAgent coordination events.

These events route between MobileAgent and child agents.
For internal agent events, see each agent's events.py file.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from llama_index.core.workflow import StopEvent
from pydantic import BaseModel

from mobilerun.agents.workflows.common.events import AgentEvent

if TYPE_CHECKING:
    from mobilerun.agents.logs.events import EventHandler


class FastAgentExecuteEvent(AgentEvent):
    instruction: str

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_execute_event(self)


class FastAgentResultEvent(AgentEvent):
    success: bool
    reason: str
    answer: str | None = None
    instruction: str

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_result_event(self)


# ============================================================================
# Manager/Executor coordination events
# ============================================================================


class ManagerInputEvent(AgentEvent):
    """Trigger Manager workflow for planning"""

    pass


class ManagerPlanEvent(AgentEvent):
    """
    Coordination event from ManagerAgent to MobileAgent.

    Used for workflow step routing only (NOT streamed to frontend).
    For internal events with memory_update metadata, see ManagerPlanDetailsEvent.
    """

    plan: str
    current_subgoal: str
    thought: str
    answer: str = ""
    success: Optional[bool] = None  # True/False if complete, None if in progress


class ExecutorInputEvent(AgentEvent):
    """Trigger Executor workflow for action execution"""

    current_subgoal: str


class ExecutorResultEvent(AgentEvent):
    """Executor finished with action result."""

    action: Dict
    outcome: bool
    error: str
    summary: str

    def log(self, out: "EventHandler") -> None:
        out.on_executor_result_event(self)


# ============================================================================
# EXTERNAL USER MESSAGE EVENTS
# ============================================================================


class ExternalUserMessageAppliedEvent(AgentEvent):
    message_ids: List[str]
    consumer: str
    step_number: int


class ExternalUserMessageDroppedEvent(AgentEvent):
    message_ids: List[str]
    reason: str
    step_number: int


# ============================================================================
# FINALIZATION EVENTS
# ============================================================================


class FinalizeEvent(AgentEvent):
    """Trigger finalization."""

    success: bool
    reason: str
    answer: str | None = None

    def log(self, out: "EventHandler") -> None:
        out.on_finalize_event(self)


class ResultEvent(StopEvent):
    """
    Final result from MobileAgent.

    Returned by MobileAgent.run() with:
    - success: Whether the task completed successfully
    - reason: Status or failure explanation
    - answer: Optional final finding from the agent
    - steps: Number of steps taken
    - structured_output: Extracted structured data (if output_model was provided)
    """

    success: bool
    reason: str
    steps: int
    answer: str | None = None
    structured_output: Optional[BaseModel] = None

    def log(self, out: "EventHandler") -> None:
        out.unknown(self)

"""
Events for the FastAgent workflow.

Internal events for streaming to frontend/logging.
"""

from typing import TYPE_CHECKING, Optional, TypedDict

from mobilerun.agents.llm.usage import UsageResult
from mobilerun.agents.workflows.common.events import AgentEvent
from mobilerun.agents.workflows.fast_agent.xml_parser import ToolCallParseStatus

if TYPE_CHECKING:
    from mobilerun.agents.logs.events import EventHandler
    from mobilerun.agents.trajectory.trajectory import Trajectory
    from mobilerun.agents.trajectory.writer import TrajectoryWriter


class FastAgentCompletion(TypedDict):
    success: bool
    reason: str
    answer: str | None
    tool_call_count: int


class FastAgentInputEvent(AgentEvent):
    """Input ready for LLM."""

    pass

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_input_event(self)


class FastAgentResponseEvent(AgentEvent):
    """LLM response received."""

    thought: str
    code: Optional[str] = None
    usage: Optional[UsageResult] = None
    tool_call_status: ToolCallParseStatus = ToolCallParseStatus.NO_MARKUP

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_response_event(self)


class FastAgentToolCallEvent(AgentEvent):
    """Tool calls ready to execute."""

    tool_calls_repr: str

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_tool_call_event(self)


class FastAgentOutputEvent(AgentEvent):
    """Tool execution result."""

    output: str

    def record(
        self,
        trajectory: "Trajectory",
        writer: "TrajectoryWriter | None",
        fast_step: int | None,
    ) -> None:
        super().record(trajectory, writer, fast_step)
        if writer is not None and fast_step is not None:
            writer.write(trajectory, stage=f"fast_agent_step_{fast_step}")

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_output_event(self)


class FastAgentEndEvent(AgentEvent):
    """FastAgent finished."""

    success: bool
    reason: str
    answer: str | None = None
    tool_call_count: int = 0

    def log(self, out: "EventHandler") -> None:
        out.on_fast_agent_end_event(self)

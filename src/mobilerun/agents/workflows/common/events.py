from typing import TYPE_CHECKING, Any, Dict

from llama_index.core.workflow import Event

if TYPE_CHECKING:
    from mobilerun.agents.logs.events import EventHandler
    from mobilerun.agents.trajectory.trajectory import Trajectory
    from mobilerun.agents.trajectory.writer import TrajectoryWriter


class AgentEvent(Event):
    def log(self, out: "EventHandler") -> None:
        out.unknown(self)

    def record(
        self,
        trajectory: "Trajectory",
        writer: "TrajectoryWriter | None",
        fast_step: int | None,
    ) -> None:
        trajectory.events.append(self)


class ScreenshotEvent(AgentEvent):
    screenshot: bytes

    def record(
        self,
        trajectory: "Trajectory",
        writer: "TrajectoryWriter | None",
        fast_step: int | None,
    ) -> None:
        trajectory.screenshot_queue.append(self.screenshot)
        trajectory.screenshot_count += 1

    def log(self, out: "EventHandler") -> None:
        out.on_screenshot_event(self)


class RecordUIStateEvent(AgentEvent):
    ui_state: list[Dict[str, Any]]

    def record(
        self,
        trajectory: "Trajectory",
        writer: "TrajectoryWriter | None",
        fast_step: int | None,
    ) -> None:
        trajectory.ui_states.append(self.ui_state)

    def log(self, out: "EventHandler") -> None:
        out.on_record_u_i_state_event(self)


class ToolExecutionEvent(AgentEvent):
    """Emitted after every tool call dispatched through ToolRegistry."""

    tool_name: str
    tool_args: Dict[str, Any]
    success: bool
    summary: str

    def log(self, out: "EventHandler") -> None:
        out.on_tool_execution_event(self)

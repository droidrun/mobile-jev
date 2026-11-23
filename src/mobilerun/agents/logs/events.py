"""
Shared event handler for CLI and SDK.

Translates workflow events into ``logging`` calls with ``extra`` params
(color, step_increment, etc.).  The actual rendering is handled by
whichever ``logging.Handler`` is attached (CLILogHandler, TUILogHandler, …).
"""

import logging
from typing import Protocol, cast

from mobilerun.logs.actions import ActionCounter, format_action

from mobilerun.agents.workflows.common.events import (
    RecordUIStateEvent,
    ScreenshotEvent,
    ToolExecutionEvent,
)
from mobilerun.agents.workflows.droid.events import (
    ExecutorResultEvent,
    FastAgentExecuteEvent,
    FastAgentResultEvent,
    FinalizeEvent,
)
from mobilerun.agents.workflows.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
)
from mobilerun.agents.workflows.fast_agent.events import (
    FastAgentEndEvent,
    FastAgentInputEvent,
    FastAgentOutputEvent,
    FastAgentResponseEvent,
    FastAgentToolCallEvent,
)
from mobilerun.agents.workflows.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)

logger = logging.getLogger("mobilerun")


class _LoggableEvent(Protocol):
    def log(self, out: "EventHandler") -> None: ...


class EventHandler:
    """Translates workflow events into logger calls.

    Counts device-action events for final summaries and converts workflow
    events into log records for the attached handler.
    """

    def __init__(self, counter: ActionCounter | None = None) -> None:
        self.counter = counter
        self.device_actions = 0

    def handle(self, event: object) -> None:
        # Llama-index and third-party events need an adapter because they do not
        # implement our logging contract. Only probe at this external boundary.
        try:
            log = cast(_LoggableEvent, event).log
        except AttributeError:
            self.unknown(event)
            return
        log(self)

    def unknown(self, event: object) -> None:
        logger.debug(f" {event.__class__.__name__}")

    def on_tool_execution_event(self, event: ToolExecutionEvent) -> None:
        if (
            self.counter is None
            and event.tool_name != "complete"
            and format_action(event.tool_name, event.tool_args)
        ):
            self.device_actions += 1

    def on_screenshot_event(self, event: ScreenshotEvent) -> None:
        logger.debug(" Taking screenshot...")

    def on_record_u_i_state_event(self, event: RecordUIStateEvent) -> None:
        logger.debug(" Recording UI state")

    def on_manager_context_event(self, event: ManagerContextEvent) -> None:
        logger.debug(" Manager preparing context...")

    def on_manager_response_event(self, event: ManagerResponseEvent) -> None:
        logger.debug(" Manager received LLM response")

    def on_manager_plan_details_event(self, event: ManagerPlanDetailsEvent) -> None:
        if event.thought:
            preview = event.thought[:120] + "..." if len(event.thought) > 120 else event.thought
            logger.debug(f" Thought: {preview}", extra={"color": "cyan"})
        if event.subgoal:
            preview = event.subgoal[:150] + "..." if len(event.subgoal) > 150 else event.subgoal
            logger.debug(f" Next step: {preview}", extra={"color": "yellow"})
        if event.answer:
            preview = event.answer[:200] + "..." if len(event.answer) > 200 else event.answer
            logger.debug(f" Answer: {preview}", extra={"color": "green"})
        if event.plan:
            logger.debug(f"▸ {event.plan}", extra={"color": "yellow"})
        if event.memory_update:
            logger.debug(
                f" Memory: {event.memory_update[:100]}...",
                extra={"color": "cyan"},
            )

    def on_executor_action_event(self, event: ExecutorActionEvent) -> None:
        if event.description:
            logger.debug(f" Action: {event.description}", extra={"color": "yellow"})
        if event.thought:
            preview = event.thought[:120] + "..." if len(event.thought) > 120 else event.thought
            logger.debug(f" Reasoning: {preview}", extra={"color": "cyan"})

    def on_executor_action_result_event(self, event: ExecutorActionResultEvent) -> None:
        if event.success:
            logger.debug(f" {event.summary}", extra={"color": "green"})
        else:
            error_msg = event.error or "Unknown error"
            logger.debug(f" {event.summary} ({error_msg})", extra={"color": "red"})

    def on_executor_result_event(self, event: ExecutorResultEvent) -> None:
        logger.debug("Step complete", extra={"color": "magenta"})

    def on_fast_agent_input_event(self, event: FastAgentInputEvent) -> None:
        logger.debug(" Task input received...")

    def on_fast_agent_response_event(self, event: FastAgentResponseEvent) -> None:
        logger.debug("FastAgent response", extra={"color": "magenta"})
        if event.thought:
            preview = event.thought[:150] + "..." if len(event.thought) > 150 else event.thought
            logger.debug(f" Thinking: {preview}", extra={"color": "cyan"})
        if event.code:
            logger.debug(" Executing action code", extra={"color": "yellow"})
            logger.debug(f"{event.code}", extra={"color": "blue"})

    def on_fast_agent_tool_call_event(self, event: FastAgentToolCallEvent) -> None:
        logger.debug(" Executing tool calls...", extra={"color": "yellow"})

    def on_fast_agent_output_event(self, event: FastAgentOutputEvent) -> None:
        if event.output:
            output = str(event.output)
            preview = output[:100] + "..." if len(output) > 100 else output
            if "Error" in output or "Exception" in output:
                logger.debug(f" Action error: {preview}", extra={"color": "red"})
            else:
                logger.debug(f" Action result: {preview}", extra={"color": "green"})

    def on_fast_agent_end_event(self, event: FastAgentEndEvent) -> None:
        status = "done" if event.success else "failed"
        color = "green" if event.success else "red"
        logger.debug(
            f"■ {status}: {event.reason} ({event.tool_call_count} runs)",
            extra={"color": color},
        )

    def on_fast_agent_execute_event(self, event: FastAgentExecuteEvent) -> None:
        logger.debug(" Starting task execution...", extra={"color": "magenta"})

    def on_fast_agent_result_event(self, event: FastAgentResultEvent) -> None:
        if event.success:
            logger.debug(f"Task result: {event.reason}", extra={"color": "green"})
        else:
            logger.debug(f"Task failed: {event.reason}", extra={"color": "red"})

    def on_finalize_event(self, event: FinalizeEvent) -> None:
        count = self.counter.count if self.counter is not None else self.device_actions
        steps = "step" if count == 1 else "steps"
        if event.success:
            logger.warning(
                f"Done in {count} {steps}",
                extra={"color": "green", "result_status": "success"},
            )
            if event.answer is not None:
                logger.warning(event.answer)
        else:
            logger.warning(
                f"Failed after {count} {steps}: {event.reason}",
                extra={"color": "red", "result_status": "failure"},
            )

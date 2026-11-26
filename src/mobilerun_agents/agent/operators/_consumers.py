import asyncio
from contextvars import Token
from pathlib import Path
from types import TracebackType
from typing import Self

from mobilerun.config.settings import get_settings
from mobilerun.macro.settings import get_macro_settings
from pydantic import BaseModel
from workflows.errors import WorkflowTimeoutError
from workflows.events import Event
from workflows.handler import WorkflowHandler

from mobilerun_agents.agent.analytics.consumer import RunAnalytics
from mobilerun_agents.agent.analytics.tracing import trace_run
from mobilerun_agents.agent.checkpoints.store import RunRecorder
from mobilerun_agents.agent.logs import ConsoleLog
from mobilerun_agents.agent.logs.diagnostics import DecisionOptions, Detail, record_decisions
from mobilerun_agents.agent.logs.redaction import redact_run_text, run_sensitive_values
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.agent.workflows.events import Invalid, Select, StepFinished, Subgoal
from mobilerun_agents.agent.workflows.planning import PlannerScreenshot, PlanningState
from mobilerun_agents.config.user_config import MobileConfig


class RunConsumers:
    def __init__(self, config: MobileConfig, run_id: str, max_steps: int) -> None:
        self.recorder: RunRecorder | None = None
        if config.checkpoints.enabled:
            folder = config.checkpoints.folder
            if folder is None:
                folder = get_settings().MOBILERUN_USER_CONFIG_DIR / "runs"
            macro_directory = Path(get_macro_settings().macro_directory).expanduser().absolute()
            self.recorder = RunRecorder(
                folder.expanduser().absolute() / run_id,
                macro_path=macro_directory / run_id / "macro.json",
                max_steps=max_steps,
            )
        self.console = ConsoleLog()
        self.analytics = RunAnalytics(config.analytics, run_id)
        self._scope = trace_run(config.tracing, run_id)
        self.steps = 0
        self._decisions_token: Token[bool] | None = None
        self._privacy_token: Token[set[str] | None] | None = None

    async def __aenter__(self) -> Self:
        self.trace = await self._scope.__aenter__()
        self._decisions_token = record_decisions.set(self.recorder is not None)
        self._privacy_token = run_sensitive_values.set(set())
        return self

    async def __aexit__(
        self, typ: type[BaseException] | None, error: BaseException | None, tb: TracebackType | None
    ) -> None:
        try:
            await self._scope.__aexit__(typ, error, tb)
        finally:
            if self._decisions_token is not None:
                record_decisions.reset(self._decisions_token)
            if self._privacy_token is not None:
                run_sensitive_values.reset(self._privacy_token)

    async def consume(self, handler: WorkflowHandler) -> tuple[AgentResult[object], str | None]:
        try:
            async for event in handler.stream_events():
                await self.on_event(event)
            result = AgentResult[object].model_validate(await handler, from_attributes=True)
            result = result.model_copy(
                update={
                    "answer": redact_run_text(result.answer),
                    "reason": redact_run_text(result.reason),
                }
            )
        except BaseException as error:
            await handler.cancel_run()
            await self.fail(error, self.steps)
            if type(error) is WorkflowTimeoutError:
                raise TimeoutError("Workflow timed out") from error
            raise
        state = await handler.ctx.store.get_state() if self.recorder and self.steps else None
        return result, await self.finish(result, state)

    async def on_event(self, event: Event) -> None:
        match event:
            case DecisionOptions():
                if self.recorder is not None:
                    path = await self.recorder.on_decision(event.choices)
                    self.console.on_event(
                        Detail(section="jev choices", text=f"{event.summary} (full list: {path})")
                    )
            case Select(snapshot=snapshot, screenshot=image):
                # Text masks cannot safely scrub arbitrary screenshot pixels.
                if run_sensitive_values.get():
                    image = None
                self.console.flush_step()
                self.analytics.on_observe(snapshot)
                if self.recorder:
                    await self.recorder.on_observe(snapshot, image, snapshot.ui_elements)
                if image:
                    self.trace.screenshot(image)
            case PlannerScreenshot(image=image):
                if not run_sensitive_values.get():
                    self.trace.screenshot(image)
            case Invalid():
                self.console.on_repair()
            case Subgoal(text=text):
                self.console.on_subgoal(text)
            case StepFinished(action=record, pre_state=pre_state):
                self.steps = record.step
                if event.display_action is not None:
                    self.console.on_action(event.display_action, delegated=event.delegated)
                if self.recorder:
                    await self.recorder.on_step(
                        record, pre_state, guard_redacted=event.guard_redacted
                    )
            case _:
                self.console.on_event(event)

    async def fail(self, error: BaseException, steps: int) -> None:
        folder = None
        match error:
            case KeyboardInterrupt() | asyncio.CancelledError() if self.recorder is not None:
                folder = str(self.recorder.folder)
        self.console.on_error(error, steps, folder)
        self.trace.finish(False, steps)
        if type(error) in (asyncio.CancelledError, KeyboardInterrupt):
            # Cancellation must unwind to the device context without waiting for a network flush.
            return
        await self.analytics.finish(
            AgentResult[object](
                success=False, reason=type(error).__name__, answer="", output=None, steps=steps
            )
        )

    async def finish(self, result: AgentResult[object], state: BaseModel | None) -> str | None:
        self.trace.finish(result.success, result.steps)
        await self.analytics.finish(result)
        path = None
        macro_path = None
        if self.recorder and state is not None:
            match state:
                case PlanningState():
                    state = state.recording_state()
            await self.recorder.finish(AgentState.model_validate(state))
            path = str(self.recorder.folder)
            macro_path = self.recorder.macro_path
        self.console.on_result(result, path, macro_path=macro_path)
        return path

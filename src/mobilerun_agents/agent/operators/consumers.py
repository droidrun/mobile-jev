"""Start one workflow and consume its only event stream."""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar
from uuid import uuid4

from mobilerun.core.devices import Device
from mobilerun.core.runs import recording_scope
from mobilerun.macro import MacroError, build_macro
from mobilerun.models.recording import RunStatus, RunStep
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel, ValidationError
from workflows import Workflow
from workflows.errors import WorkflowTimeoutError
from workflows.events import Event
from workflows.resource import ResourceManager

from mobilerun_agents.agent.analytics.consumer import RunAnalytics
from mobilerun_agents.agent.analytics.tracing import trace_run
from mobilerun_agents.agent.events import Acted, Finished, Observed, Selected, Subgoal
from mobilerun_agents.agent.logs import ConsoleLog
from mobilerun_agents.agent.logs.memory import MemoryDetail, memory_evidence_scope
from mobilerun_agents.agent.memory.memory import _RunMemory
from mobilerun_agents.agent.memory.registry import block_tools
from mobilerun_agents.agent.tools import Tool
from mobilerun_agents.agent.tools.registry import ToolRegistry, build_registry
from mobilerun_agents.agent.workflows.core.privacy import mask_value
from mobilerun_agents.agent.workflows.core.state import AgentState, Outcome, PlanningState
from mobilerun_agents.agent.workflows.resources import Services, run_resources
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import MobileConfig

from .evidence import AgentEntry, AgentEntryKind, AgentEvidence
from .recording import resolve_recording

T = TypeVar("T", bound=BaseModel)
_log = logging.getLogger(__name__)


class WorkflowFactory(Protocol):
    def __call__(self, timeout: float | None, resource_manager: ResourceManager) -> Workflow: ...


async def execute_workflow(
    factory: WorkflowFactory,
    config: MobileConfig,
    device: Device,
    goal: str,
    tools: Sequence[Tool],
    output: type[T] | None,
    max_steps: int,
    timeout: float | None,
    memory_factory: Callable[[], _RunMemory],
    planning: bool = False,
    record: bool | None = None,
) -> ExecutionResult[T]:
    device, recorder, owned = resolve_recording(device, record, config)
    status: RunStatus = "failed"
    with recording_scope(enabled=recorder is not None, by="agent"):
        try:
            folder = recorder.start_run() if recorder is not None else None
            memory = memory_factory()
            try:
                registry = build_registry(
                    device,
                    tools if planning else [*tools, *block_tools(memory.memory_blocks)],
                    config.tools,
                    None if planning else output,
                    config.agent.decision,
                )
                manager = await run_resources(device, memory, Services(registry))
                workflow = factory(timeout=timeout, resource_manager=manager)
                result = await _consume(
                    config, workflow, folder, device, goal, output, max_steps, registry
                )
            finally:
                try:
                    await memory.aclose()
                except Exception:
                    _log.warning("Unable to close run memory")
            if result.success:
                status = "succeeded"
        except (asyncio.CancelledError, KeyboardInterrupt):
            status = "cancelled"
            raise
        finally:
            if owned and recorder is not None:
                recorder.finish(status)
        if owned and status == "succeeded" and folder is not None:
            _export(folder)
        return result


def _export(folder: Path) -> None:
    path = folder / "steps.jsonl"
    try:
        with path.open(encoding="utf-8") as file:
            has_success = any(RunStep.model_validate_json(line).ok for line in file)
        if has_success:
            build_macro(path, folder.name)
    except (OSError, ValidationError, MacroError):
        # A completed agent can have no replayable steps (e.g. all masked).
        _log.debug("No macro published from the recorded agent run")


async def _consume(
    config: MobileConfig,
    workflow: Workflow,
    folder: Path | None,
    device: Device,
    goal: str,
    output: type[T] | None,
    max_steps: int,
    registry: ToolRegistry,
) -> ExecutionResult[T]:
    run_id = folder.name if folder is not None else uuid4().hex
    evidence = AgentEvidence(folder) if folder is not None else None
    sensitive = False
    redact_code = device.known_codes_redactor()

    def redact(text: str) -> str:
        if text and (sensitive or registry.sensitive_seen or device._extension_state.sensitive_ui):
            return "[omitted]"
        return redact_code(str(mask_value(text)))

    console = ConsoleLog(redact)
    analytics = RunAnalytics(config.analytics, run_id)
    finished: Finished | None = None
    pending: list[Event] = []
    batch: list[Acted] = []
    steps = 0
    action_steps = 0
    shared_steps = 0

    def display_message(message: str | None, success: bool, reason: str) -> str:
        if message is None:
            message = render_prompt(config.agent.decision.completion_prompt, success=success)
        return message or reason

    def append(kind: AgentEntryKind, text: str, step_ids: list[int]) -> None:
        if evidence is not None:
            evidence.append(AgentEntry(kind=kind, text=redact_code(text), step_ids=step_ids))

    def flush(step_ids: list[int]) -> None:
        for detail in pending:
            console.event(detail)
            match detail:
                case Observed(step=step, package=package, activity=activity):
                    analytics.on_observe(package, activity, step)
                case Selected(thought=thought) if thought:
                    append("thought", redact(thought), step_ids)
                case Subgoal(text=text):
                    append("plan", redact(text), step_ids)
                case MemoryDetail(text=text):
                    append("memory", redact(text), step_ids)
        pending.clear()

    def flush_actions() -> None:
        if not batch:
            return
        flush(list(dict.fromkeys(i for event in batch for i in event.step_ids)))
        for acted in batch:
            console.event(acted)
            append("tool", str(mask_value(acted.action.call.tool_name)), acted.step_ids)
        batch.clear()

    async with (
        memory_evidence_scope(evidence is not None),
        trace_run(config.tracing, run_id) as trace,
    ):
        handler = workflow.run(goal=goal, max_steps=max_steps)

        async def consume_events() -> None:
            nonlocal action_steps, finished, sensitive, shared_steps, steps
            async for event in handler.stream_events():
                match event:
                    case Acted(action=action):
                        batch.append(event)
                        if action_steps < action.step:
                            action_steps = steps = action.step
                        continue
                flush_actions()
                match event:
                    case Selected(calls=calls):
                        sensitive |= any(
                            registry.is_sensitive_choice(call.tool_name) for call in calls
                        )
                        pending.append(event)
                    case Finished():
                        finished = event
                    case Subgoal(result=subgoal, step=step) if subgoal is not None:
                        if step is not None:
                            shared_steps = max(shared_steps, step)
                        flush([])
                        console.subgoal_finished(
                            subgoal.success,
                            display_message(subgoal.message, subgoal.success, subgoal.reason),
                        )
                    case Subgoal(step=step) if step is not None:
                        shared_steps = max(shared_steps, step)
                        pending.append(event)
                    case _:
                        pending.append(event)
            flush_actions()

        events = asyncio.create_task(consume_events())
        try:
            await asyncio.shield(events)
            outcome = Outcome.model_validate(await handler)
            if finished is None:
                raise RuntimeError("Workflow ended without a Finished event")
            steps = finished.steps
            shared_steps = max(shared_steps, finished.steps)
            flush([])
            answer = display_message(outcome.message, outcome.success, outcome.reason)
            public_output = None
            if output is not None and outcome.success:
                try:
                    public_output = output.model_validate_json(
                        json.dumps(outcome.output), strict=True
                    )
                except ValidationError:
                    _log.warning("Typed output omitted: output did not match the schema")
            result: ExecutionResult[T] = ExecutionResult(
                success=outcome.success,
                answer=answer,
                output=public_output,
                steps=steps,
                device=device.identifier or device.platform,
                trajectory_path=str(folder) if folder else None,
            )
            console.finish(outcome.success, answer, finished, folder)
            trace.finish(outcome.success, steps)
            await analytics.finish(outcome.success, steps)
            return result
        except BaseException as error:
            if not handler.is_done():
                await handler.cancel_run()
            if not events.done():
                await asyncio.wait({events}, timeout=1)
                if not events.done():
                    events.cancel()
            await asyncio.gather(events, return_exceptions=True)
            flush_actions()
            cancelled = type(error) in (asyncio.CancelledError, KeyboardInterrupt)
            trace.finish(False, steps)
            console_steps = shared_steps or steps
            try:
                match await handler.ctx.store.get_state():
                    case PlanningState(planner_turns=turns, executor_steps=executed):
                        console_steps = turns + max(executed, action_steps)
                    case AgentState():
                        console_steps = steps
            except BaseException:
                # State is best-effort here; preserve the error that ended the run.
                pass
            console.finish(
                False,
                "cancelled" if cancelled else type(error).__name__,
                console_steps,
                folder,
            )
            if not cancelled:
                await analytics.finish(False, steps)
            if type(error) is WorkflowTimeoutError:
                raise TimeoutError("Workflow timed out") from error
            raise

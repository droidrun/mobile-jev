"""Start one workflow and consume its only event stream."""

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar
from uuid import uuid4

from mobilerun.core.devices import Device
from mobilerun.macro.settings import get_macro_settings
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError
from workflows import Context, Workflow
from workflows.errors import WorkflowTimeoutError
from workflows.events import Event
from workflows.resource import ResourceManager

from mobilerun_agents.agent.analytics.consumer import RunAnalytics
from mobilerun_agents.agent.analytics.tracing import trace_run
from mobilerun_agents.agent.checkpoints.macro_export import macro_data
from mobilerun_agents.agent.checkpoints.store import save_context, save_macro
from mobilerun_agents.agent.events import Acted, Finished, Observed, Subgoal
from mobilerun_agents.agent.logs import ConsoleLog
from mobilerun_agents.agent.memory.memory import build_run_memory
from mobilerun_agents.agent.tools import Tool
from mobilerun_agents.agent.tools.helpers.images import ImageStore
from mobilerun_agents.agent.tools.registry import ToolRegistry, build_registry
from mobilerun_agents.agent.workflows.core.state import AgentState, Outcome, PlanningState
from mobilerun_agents.agent.workflows.resources import Services, run_resources
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.settings import get_agent_settings
from mobilerun_agents.config.user_config import MobileConfig

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
    typed_completion: bool = True,
) -> ExecutionResult[T]:
    run_id = uuid4().hex
    images = ImageStore(run_id, keep=config.checkpoints.enabled)
    try:
        registry = build_registry(
            device, tools, config.tools, output if typed_completion else None, config.agent.decision
        )
        memory = build_run_memory(config.memory, device.platform, config.agent)
        try:
            manager = await run_resources(device, memory, Services(registry, images))
            workflow = factory(timeout=timeout, resource_manager=manager)
            return await _consume(
                config, workflow, registry, images, run_id, device, goal, output, max_steps
            )
        finally:
            try:
                await memory.aclose()
            except Exception:
                _log.warning("Unable to close run memory")
    finally:
        images.close()


async def _consume(
    config: MobileConfig,
    workflow: Workflow,
    registry: ToolRegistry,
    images: ImageStore,
    run_id: str,
    device: Device,
    goal: str,
    output: type[T] | None,
    max_steps: int,
) -> ExecutionResult[T]:
    folder = get_agent_settings().runs_dir / run_id if config.checkpoints.enabled else None
    checkpoint = folder / "context.json" if folder is not None else None
    redact = registry.redactor
    console = ConsoleLog(redact, folder)
    analytics = RunAnalytics(config.analytics, run_id)
    finished: Finished | None = None
    pending: list[Event] = []
    steps = 0
    action_steps = 0
    shared_steps = 0
    checkpoint_revision = 0

    def display_message(message: str | None, success: bool, reason: str) -> str:
        if message is None:
            message = render_prompt(config.agent.decision.completion_prompt, success=success)
        return redact.text(message or reason)

    def flush() -> None:
        for detail in pending:
            console.event(detail)
            match detail:
                case Observed(step=step, package=package, activity=activity):
                    analytics.on_observe(
                        redact.text(package) if package else None,
                        redact.text(activity) if activity else None,
                        step,
                    )
        pending.clear()

    async with trace_run(config.tracing, run_id) as trace:
        handler = workflow.run(goal=goal, max_steps=max_steps)
        try:
            async for event in handler.stream_events():
                match event:
                    case Acted(action=action):
                        flush()
                        console.event(event)
                        if action_steps < action.step:
                            action_steps = steps = action.step
                            if checkpoint is not None:
                                revision = redact.revision
                                await save_context(checkpoint, handler.ctx, redact)
                                checkpoint_revision = revision
                    case Finished():
                        finished = event
                    case Subgoal(result=subgoal, step=step) if subgoal is not None:
                        if step is not None:
                            shared_steps = max(shared_steps, step)
                        flush()
                        console.subgoal_finished(
                            subgoal.success,
                            display_message(subgoal.message, subgoal.success, subgoal.reason),
                        )
                    case Subgoal(step=step) if step is not None:
                        shared_steps = max(shared_steps, step)
                        pending.append(event)
                    case _:
                        pending.append(event)
            outcome = Outcome.model_validate(await handler)
            if finished is None:
                raise RuntimeError("Workflow ended without a Finished event")
            steps = finished.steps
            shared_steps = max(shared_steps, finished.steps)
            if checkpoint is not None:
                revision = redact.revision
                saved = await save_context(checkpoint, handler.ctx, redact)
                checkpoint_revision = revision
                state = await Context.from_dict(workflow, saved).store.get_state()
                match state:
                    case AgentState() | PlanningState():
                        macro_path = (
                            Path(get_macro_settings().macro_directory).expanduser()
                            / run_id
                            / "macro.json"
                        )
                        await save_macro(macro_path, macro_data(state.goal, state.actions, redact))
            if (
                config.tracing.enabled
                and config.tracing.langfuse_screenshots
                and not redact.revision
            ):
                for path in sorted(images.folder.glob("*.png")):
                    trace.screenshot(await asyncio.to_thread(path.read_bytes))
            flush()
            answer = display_message(outcome.message, outcome.success, outcome.reason)
            public_output = None
            if output is not None and outcome.success:
                try:
                    public_output = output.model_validate_json(
                        json.dumps(redact.value(outcome.output)), strict=True
                    )
                except ValidationError:
                    pass
                else:
                    # Defaults and validators can reintroduce protected data after input masking.
                    try:
                        serialized = json.loads(public_output.model_dump_json(by_alias=True))
                    except PydanticSerializationError:
                        # Without a successful safety check, do not expose the model.
                        public_output = None
                    else:
                        if redact.value(serialized) != serialized:
                            public_output = None
                if public_output is None:
                    _log.warning("Typed output omitted: redacted output did not match the schema")
            result: ExecutionResult[T] = ExecutionResult(
                success=outcome.success,
                answer=answer,
                output=public_output,
                steps=steps,
                device=redact.text(device.identifier or device.platform),
                trajectory_path=redact.text(str(folder)) if folder else None,
            )
            console.finish(outcome.success, answer, finished, folder)
            trace.finish(outcome.success, steps)
            await analytics.finish(outcome.success, steps)
            return result
        except BaseException as error:
            if not handler.is_done():
                await handler.cancel_run()
            if (
                checkpoint is not None
                and checkpoint.is_file()
                and checkpoint_revision != redact.revision
            ):
                try:
                    await save_context(checkpoint, handler.ctx, redact)
                except Exception:
                    # The earlier checkpoint can contain a newly discovered secret.
                    checkpoint.unlink(missing_ok=True)
                    _log.warning("Unable to refresh checkpoint with new redactions")
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

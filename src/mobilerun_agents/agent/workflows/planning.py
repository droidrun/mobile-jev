"""Plan the next subgoal, then execute it in an independent agent workflow."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Annotated, Any

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.macro.models import build_screen_snapshot
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.handler import WorkflowHandler
from workflows.resource import Resource
from workflows.retry_policy import retry_if_exception_type, retry_policy

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.logs.diagnostics import (
    ActionDetail,
    DecisionOptions,
    Detail,
    ModelCall,
    Screen,
    private_values,
    sink,
)
from mobilerun_agents.agent.logs.redaction import (
    redact_run_arguments,
    redact_run_text,
    redact_run_value,
    run_sensitive_values,
)
from mobilerun_agents.agent.planner import Finish, Plan, Planner, PlanningInput, SubgoalResult
from mobilerun_agents.agent.tools.base import Tool
from mobilerun_agents.config.user_config import AgentConfig, ToolsConfig

from .agent import AgentWorkflow, Begin, TransientSelectionError, validate_output
from .core.result import AgentResult
from .core.state import ActionRecord, AgentState
from .events import Act, Invalid, Select, StepFinished, Subgoal


class PlanningState(BaseModel):
    goal: str = ""
    max_steps: int = 0
    remaining_plan: list[str] = Field(default_factory=list)
    results: list[SubgoalResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    planner_turns: int = 0
    step: int = 0
    actions: list[ActionRecord] = Field(default_factory=list)
    repair_attempts: int = 0
    output_attempts: int = 0
    repair: str | None = None

    def recording_state(self) -> AgentState:
        return AgentState(
            goal=self.goal, max_steps=self.max_steps, step=self.step, actions=self.actions
        )


class RunSubgoal(Event):
    goal: str


class SubgoalDone(Event):
    result: SubgoalResult
    notes: list[str]
    actions: list[ActionRecord]


@dataclass(frozen=True)
class _PlanningResources:
    device: Device
    tools: tuple[Tool, ...]
    tool_config: ToolsConfig
    config: AgentConfig
    output: type[BaseModel] | None
    memory_factory: Callable[[], Memory] | None


_active_planning: ContextVar[_PlanningResources] = ContextVar("planning_resources")


def _get_resources() -> _PlanningResources:
    return _active_planning.get()


_Resources = Annotated[_PlanningResources, Resource(_get_resources, cache=False)]


def _stop(
    state: PlanningState,
    reason: str,
    success: bool = False,
    answer: str = "",
    output: BaseModel | None = None,
) -> StopEvent:
    return StopEvent(
        result=AgentResult[Any](
            success=success, reason=reason, answer=answer, output=output, steps=state.step
        )
    )


def _offset_record(record: ActionRecord, offset: int) -> ActionRecord:
    args = redact_run_arguments(record.args)
    name = redact_run_text(record.name)
    return record.model_copy(
        update={
            "step": record.step + offset,
            "name": name,
            "args": args,
            "summary": redact_run_text(record.summary),
            "replayable": record.replayable and name == record.name and args == record.args,
        }
    )


def _optional_text(text: str | None) -> str | None:
    return redact_run_text(text) if text is not None else None


def _stream_event(event: Event) -> Event:
    """Mask textual payloads, keeping event types and binary screenshot data intact."""
    match event:
        case Subgoal(text=text) | Detail(text=text):
            return event.model_copy(update={"text": redact_run_text(text)})
        case Screen(package=package, activity=activity):
            return event.model_copy(
                update={"package": _optional_text(package), "activity": _optional_text(activity)}
            )
        case ModelCall(name=name):
            return event.model_copy(update={"name": redact_run_text(name)})
        case Invalid(error=error, diagnostic=diagnostic):
            return event.model_copy(
                update={"error": redact_run_text(error), "diagnostic": redact_run_text(diagnostic)}
            )
        case DecisionOptions(summary=summary, choices=choices):
            return event.model_copy(
                update={
                    "summary": redact_run_text(summary),
                    "choices": [redact_run_arguments(c) for c in choices],
                }
            )
        case ActionDetail(wire=wire, registry=registry, args=args, summary=summary):
            return event.model_copy(
                update={
                    "wire": redact_run_text(wire),
                    "registry": redact_run_text(registry),
                    "args": redact_run_arguments(args),
                    "summary": redact_run_text(summary),
                }
            )
        case Act(selection=selection):
            return event.model_copy(
                update={
                    "selection": replace(
                        selection,
                        actions=[
                            replace(
                                call,
                                name=redact_run_text(call.name),
                                parameters=redact_run_arguments(call.parameters),
                                error=_optional_text(call.error),
                                call_id=None,
                            )
                            for call in selection.actions
                        ],
                        thought=redact_run_text(selection.thought),
                        description=redact_run_text(selection.description),
                        memory_update=redact_run_text(selection.memory_update),
                        response_text=redact_run_text(selection.response_text),
                        parse_error=_optional_text(selection.parse_error),
                        failure=_optional_text(selection.failure),
                        response=None,
                        parse_result=None,
                    )
                }
            )
    return event


class PlanningWorkflow(Workflow):
    def __init__(self, planner: Planner, executor: ActionSelector, **kw: Any) -> None:
        super().__init__(**kw)
        self.planner = planner
        self.executor = executor

    def run(
        self,
        ctx: Context[Any] | None = None,
        start_event: StartEvent | None = None,
        device: Device | None = None,
        tools: tuple[Tool, ...] | list[Tool] = (),
        tool_config: ToolsConfig | None = None,
        config: AgentConfig | None = None,
        output: type[BaseModel] | None = None,
        memory_factory: Callable[[], Memory] | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        if device is None:
            raise ValueError("A device is required")
        config = config or AgentConfig()
        if start_event is None:
            kwargs.setdefault("max_steps", config.planning_max_steps)
        token = _active_planning.set(
            _PlanningResources(
                device, tuple(tools), tool_config or ToolsConfig(), config, output, memory_factory
            )
        )
        # Share the same mutable set with every nested AgentWorkflow and the
        # operator's recorder. Child updates become visible before forwarding.
        shared = run_sensitive_values.get()
        privacy = run_sensitive_values.set(shared if shared is not None else set())
        try:
            return super().run(ctx=ctx, start_event=start_event, **kwargs)
        finally:
            run_sensitive_values.reset(privacy)
            _active_planning.reset(token)

    @step(retry_policy=retry_policy(retry=retry_if_exception_type(TransientSelectionError)))
    async def plan(
        self, ctx: Context[PlanningState], ev: Begin | SubgoalDone, run: _Resources
    ) -> RunSubgoal | StopEvent:
        state = await ctx.store.get_state()
        # A transport retry re-enters the step with the same event. The already
        # saved child result and budgets must not be applied or reset again.
        if ctx.retry_info().retry_number == 0:
            match ev:
                case Begin(goal=goal, max_steps=max_steps):
                    state = PlanningState(goal=goal, max_steps=max_steps)
                case SubgoalDone(result=result, notes=notes, actions=actions):
                    state.results.append(result)
                    state.step += result.steps
                    state.actions.extend(actions)
                    state.notes = list(dict.fromkeys([*state.notes, *notes]))

        if state.step >= state.max_steps:
            await ctx.store.set_state(state)
            return _stop(state, "max_steps")
        if state.planner_turns >= run.config.planning.max_planner_turns:
            await ctx.store.set_state(state)
            return _stop(state, "max_planner_turns")

        ui = await run.device.state.refresh()
        debug = private_values.get() is not None

        def emit(event: Event) -> None:
            ctx.write_event_to_stream(_stream_event(event))

        if debug:
            emit(
                Screen(
                    package=ui.phone_state.get("packageName"),
                    activity=ui.phone_state.get("activityName"),
                    elements=len(ui.elements),
                )
            )
        token = sink.set(emit if debug else None)
        try:
            while state.planner_turns < run.config.planning.max_planner_turns:
                state.planner_turns += 1
                await ctx.store.set_state(state)
                if debug:
                    mark = ("✓" if state.results[-1].success else "✗") if state.results else "—"
                    emit(
                        Detail(
                            section="planning",
                            text=f"turn {state.planner_turns} · remaining plan {len(state.remaining_plan)} · last result {mark}",
                        )
                    )
                request = PlanningInput(
                    goal=state.goal,
                    config=run.config,
                    remaining_plan=tuple(state.remaining_plan),
                    results=tuple(r.model_copy(deep=True) for r in state.results),
                    notes=tuple(state.notes),
                    package=ui.phone_state.get("packageName"),
                    activity=ui.phone_state.get("activityName"),
                    ui=ui.formatted_text,
                    output=run.output,
                    repair=state.repair,
                )
                match await self.planner.plan(request):
                    case Plan(subgoals=()):
                        state.repair = "set_plan requires a non-empty list of subgoals"
                        emit(Invalid(error=state.repair))
                        state.repair_attempts += 1
                        if (
                            state.repair_attempts
                            >= run.config.decision.max_response_repair_attempts
                        ):
                            await ctx.store.set_state(state)
                            return _stop(state, "repair_attempts")
                    case Plan(subgoals=(first, *rest)):
                        state.remaining_plan = rest
                        state.repair = None
                        state.repair_attempts = 0
                        await ctx.store.set_state(state)
                        if debug:
                            emit(
                                Detail(
                                    section="planning",
                                    text=f"subgoals: {redact_run_text('; '.join([first, *rest]))[:290]}",
                                )
                            )
                        return RunSubgoal(goal=first)
                    case Finish(success=success, message=message, output=output):
                        try:
                            value = validate_output(run.output, output) if success else None
                            if value is not None:
                                payload = value.model_dump(mode="json")
                                if redact_run_value(payload) != payload:
                                    raise ValueError("Planner output contains protected values")
                        except (TypeError, ValueError):
                            state.output_attempts += 1
                            state.repair = "Planner output is invalid. Correct the output to match the `complete` schema."
                            emit(Invalid(error=state.repair, diagnostic="Invalid planner output"))
                            if state.output_attempts >= run.config.max_output_validation_attempts:
                                await ctx.store.set_state(state)
                                return _stop(state, "output_attempts")
                        else:
                            state.repair = None
                            await ctx.store.set_state(state)
                            return _stop(
                                state,
                                "complete",
                                success=success,
                                answer=redact_run_text(message),
                                output=value,
                            )
        except (TimeoutError, ConnectionError) as exc:
            raise TransientSelectionError("Model temporarily unavailable") from exc
        except InvalidResponse as exc:
            emit(Invalid(error=redact_run_text(str(exc)), diagnostic="Invalid planner tool call"))
            await ctx.store.set_state(state)
            return _stop(state, "repair_attempts")
        finally:
            sink.reset(token)
        await ctx.store.set_state(state)
        return _stop(state, "max_planner_turns")

    @step
    async def execute(
        self, ctx: Context[PlanningState], ev: RunSubgoal, run: _Resources
    ) -> SubgoalDone:
        state = await ctx.store.get_state()
        ctx.write_event_to_stream(_stream_event(Subgoal(text=ev.goal)))
        if private_values.get() is not None:
            ctx.write_event_to_stream(
                _stream_event(
                    Detail(section="planning", text=f"executor: {self.executor.diagnostic_name}")
                )
            )
        child = AgentWorkflow(self.executor, timeout=None).run(
            device=run.device,
            tools=list(run.tools),
            tool_config=run.tool_config,
            config=run.config,
            memory=run.memory_factory() if run.memory_factory else None,
            goal=ev.goal,
            max_steps=min(run.config.planning.subgoal_steps, state.max_steps - state.step),
        )
        actions = []
        ui_elements: list[dict[str, JSONValue]] = []
        try:
            async for event in child.stream_events():
                match event:
                    case StopEvent():
                        # A streamed stop event would end the parent stream prematurely.
                        continue
                    case Select(snapshot=snapshot):
                        ui_elements = snapshot.ui_elements
                        event = event.model_copy(
                            update={
                                "snapshot": snapshot.model_copy(
                                    update={
                                        "step": snapshot.step + state.step,
                                        "goal": redact_run_text(snapshot.goal),
                                        "notes": [redact_run_text(n) for n in snapshot.notes],
                                        "package": _optional_text(snapshot.package),
                                        "activity": _optional_text(snapshot.activity),
                                        "ui_elements": [
                                            redact_run_arguments(e) for e in ui_elements
                                        ],
                                        "device_context": [
                                            replace(
                                                item,
                                                source=redact_run_text(item.source),
                                                kind=redact_run_text(item.kind),
                                                data=redact_run_arguments(dict(item.data)),
                                            )
                                            for item in snapshot.device_context
                                        ],
                                        "last_actions": [
                                            _offset_record(a, state.step)
                                            for a in snapshot.last_actions
                                        ],
                                    }
                                )
                            }
                        )
                    case StepFinished(action=action, display_action=display, pre_state=pre_state):
                        record = _offset_record(action, state.step)
                        actions.append(record)
                        guard = (
                            pre_state
                            if pre_state is not None
                            else build_screen_snapshot(ui_elements)
                        )
                        masked = redact_run_arguments(guard)
                        event = event.model_copy(
                            update={
                                "action": record,
                                "pre_state": masked,
                                "guard_redacted": masked != guard,
                                "display_action": (
                                    _offset_record(display, state.step) if display else None
                                ),
                            }
                        )
                ctx.write_event_to_stream(_stream_event(event))
            result: AgentResult[Any] = await child
            child_state = AgentState.model_validate(await child.ctx.store.get_state())
        finally:
            if not child.is_done():
                await child.cancel_run()
        return SubgoalDone(
            result=SubgoalResult(
                goal=redact_run_text(ev.goal),
                success=result.success,
                message=result.answer
                or ("step limit reached" if result.reason == "max_steps" else result.reason),
                steps=result.steps,
            ),
            notes=child_state.notes,
            actions=actions,
        )

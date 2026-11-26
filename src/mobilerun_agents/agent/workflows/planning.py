"""Plan the next subgoal, then execute it in an independent agent workflow."""

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.errors import DeviceOperationNotSupported
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from workflows.handler import WorkflowHandler
from workflows.retry_policy import retry_if_exception_type, retry_policy

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.logs.diagnostics import (
    ActionDetail,
    DecisionOptions,
    Detail,
    ModelCall,
    Retry,
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
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig, DecisionConfig, ToolsConfig

from .agent import (
    AgentWorkflow,
    Begin,
    TransientSelectionError,
    public_record,
    validate_output,
)
from .core.result import AgentResult, FailureSummary
from .core.state import ActionRecord, AgentState
from .events import Act, Invalid, Observed, StepFinished, Subgoal
from .resources import DeviceResource, active_device, record_offset, run_image_scope, trace_image


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
    subgoal: str | None = None

    def recording_state(self) -> AgentState:
        return AgentState(
            goal=self.goal, max_steps=self.max_steps, step=self.step, actions=self.actions
        )


class RunSubgoal(Event):
    pass


class SubgoalDone(Event):
    result: SubgoalResult
    actions: list[ActionRecord]
    notes: list[str]


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
    number = record.step + offset
    return record.model_copy(update={"step": number})


def _optional_text(text: str | None) -> str | None:
    return redact_run_text(text) if text is not None else None


_FAILURE_REASONS = frozenset({"max_steps", "stuck", "cycle"})
_BLOCKER_TOPIC = re.compile(
    r"\b(?:location|region|country|destination|sign[ -]?in|log[ -]?in|filter(?:s|ed)?|permissions?|"
    r"pop[ -]?ups?|modal|dialog|deliver(?:y|ed)|ship(?:ping)?)\b",
    re.IGNORECASE,
)
_BLOCKER_SIGNAL = re.compile(
    r"\b(?:can't|cannot|couldn't|doesn't|won't|not|must|please|need(?:s|ed)?|"
    r"block(?:s|ed|ing)?|hid(?:es|den|ing)|requir(?:e[sd]?|ing)|"
    r"denied|unavailable|exclud(?:es?|ed|ing)|dismiss(?:ed|ing)?|"
    r"prevent(?:s|ed|ing)?|obscur(?:es|ed|ing))\b",
    re.IGNORECASE,
)


def _blockers(thoughts: Sequence[str]) -> tuple[str, ...]:
    evidence: dict[str, None] = {}
    for thought in thoughts:
        for sentence in re.split(r"\r?\n|(?<=[.!?])\s+", thought.strip()):
            text = sentence.strip()
            if text.lower().startswith("blocked:") or (
                _BLOCKER_TOPIC.search(text) and _BLOCKER_SIGNAL.search(text)
            ):
                evidence[text] = None
    return tuple(evidence)[:3]


def _needs_observation(result: SubgoalResult, config: DecisionConfig) -> bool:
    message = result.message.strip()
    return not message or message in {
        render_prompt(config.completion_prompt, success=True),
        render_prompt(config.completion_prompt, success=False),
        render_prompt(config.criteria_prompt, kind="complete_success"),
        render_prompt(config.criteria_prompt, kind="complete_failure"),
    }


def _summarize_failure(reason: str, state: AgentState) -> FailureSummary:
    thoughts = list(dict.fromkeys(a.thought for a in state.actions if a.thought))
    thoughts.extend(state.pending_thoughts)

    def clip(text: str, limit: int) -> str:
        return f"{text[: limit - 1]}…" if len(text) > limit else text

    return FailureSummary(
        reason=reason,
        thoughts=tuple(clip(thought, 240) for thought in thoughts[-3:]),
        blockers=tuple(clip(blocker, 240) for blocker in _blockers(thoughts)),
        screens=tuple(state.screens.values()),
        actions=tuple(
            clip(f"{action.name} ({'ok' if action.ok else 'failed'}): {action.summary}", 180)
            for action in state.actions[-3:]
        ),
    )


def _stream_event(event: Event) -> Event:
    """Mask public event copies without modifying workflow state or model inputs."""
    match event:
        case Subgoal(text=text) | Detail(text=text):
            return event.model_copy(update={"text": redact_run_text(text)})
        case Screen(package=package, activity=activity):
            return event.model_copy(
                update={"package": _optional_text(package), "activity": _optional_text(activity)}
            )
        case ModelCall(name=name):
            return event.model_copy(update={"name": redact_run_text(name)})
        case Retry(reason=reason):
            return event.model_copy(update={"reason": redact_run_text(reason)})
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
    _config: AgentConfig
    _tool_config: ToolsConfig
    _tools: list[Tool]
    _output: type[BaseModel] | None
    _memory_factory: Callable[[], Memory] | None

    def __init__(self, planner: Planner, executor: ActionSelector, **kw: Any) -> None:
        super().__init__(**kw)
        # An injected manager belongs to the template, never an invocation.
        self._workflow_options = {
            key: value for key, value in kw.items() if key != "resource_manager"
        }
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
        workflow = type(self)(self.planner, self.executor, **self._workflow_options)
        workflow._tools = list(tools)
        workflow._tool_config = tool_config or ToolsConfig()
        workflow._config = config
        workflow._output = output
        workflow._memory_factory = memory_factory
        token = active_device.set(device)
        # Share the same mutable set with every nested AgentWorkflow and the
        # operator's recorder. Child updates become visible before forwarding.
        shared = run_sensitive_values.get()
        privacy = run_sensitive_values.set(shared if shared is not None else set())
        try:
            with run_image_scope() as images:
                handler = Workflow.run(workflow, ctx=ctx, start_event=start_event, **kwargs)
                if images is not None:
                    handler._result_task.add_done_callback(lambda _: images.close())
                return handler
        finally:
            run_sensitive_values.reset(privacy)
            active_device.reset(token)

    @step(retry_policy=retry_policy(retry=retry_if_exception_type(TransientSelectionError)))
    async def plan(
        self, ctx: Context[PlanningState], ev: Begin | SubgoalDone, device: DeviceResource
    ) -> RunSubgoal | StopEvent:
        state = await ctx.store.get_state()
        # A transport retry re-enters the step with the same event. The already
        # saved child result and budgets must not be applied or reset again.
        if ctx.retry_info().retry_number == 0:
            match ev:
                case Begin(goal=goal, max_steps=max_steps):
                    state = PlanningState(goal=goal, max_steps=max_steps)
                case SubgoalDone(result=result, actions=actions, notes=notes):
                    state.results.append(result)
                    state.step += result.steps
                    state.actions.extend(actions)
                    state.notes = list(dict.fromkeys([*state.notes, *notes]))
                    state.subgoal = None

        if state.step >= state.max_steps:
            await ctx.store.set_state(state)
            return _stop(state, "max_steps")
        if state.planner_turns >= self._config.planning.max_planner_turns:
            await ctx.store.set_state(state)
            return _stop(state, "max_planner_turns")

        ui = await device.state.refresh()
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
            observe_screen = bool(
                state.results and _needs_observation(state.results[-1], self._config.decision)
            )
            screenshot = None
            capture_screen = observe_screen and self.planner.supports_images
            while state.planner_turns < self._config.planning.max_planner_turns:
                state.planner_turns += 1
                await ctx.store.set_state(state)
                if capture_screen and screenshot is None:
                    try:
                        screenshot = await device.screenshot()
                    except DeviceOperationNotSupported:
                        capture_screen = False
                    # Planner images are not the pre-action observations used by checkpoints.
                    if (
                        screenshot
                        and not run_sensitive_values.get()
                        and (capture := trace_image.get()) is not None
                    ):
                        capture(screenshot)
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
                    config=self._config,
                    remaining_plan=tuple(state.remaining_plan),
                    results=tuple(result.model_copy(deep=True) for result in state.results),
                    notes=tuple(state.notes),
                    package=ui.phone_state.get("packageName"),
                    activity=ui.phone_state.get("activityName"),
                    ui=ui.formatted_text,
                    output=self._output,
                    repair=state.repair,
                    observe_screen=observe_screen,
                    screenshot=screenshot,
                )
                match await self.planner.plan(request):
                    case Plan(subgoals=()):
                        state.repair = "set_plan requires a non-empty list of subgoals"
                        emit(Invalid(error=state.repair))
                        state.repair_attempts += 1
                        if (
                            state.repair_attempts
                            >= self._config.decision.max_response_repair_attempts
                        ):
                            await ctx.store.set_state(state)
                            return _stop(state, "repair_attempts")
                    case Plan(subgoals=(first, *rest)):
                        state.subgoal = first
                        state.remaining_plan = rest
                        state.repair = None
                        state.repair_attempts = 0
                        await ctx.store.set_state(state)
                        if debug:
                            subgoals = "; ".join(redact_run_text(goal) for goal in (first, *rest))
                            emit(
                                Detail(
                                    section="planning",
                                    text=f"subgoals: {subgoals[:290]}",
                                )
                            )
                        return RunSubgoal()
                    case Finish(success=success, message=message, output=output):
                        try:
                            value = validate_output(self._output, output) if success else None
                            if value is not None:
                                payload = value.model_dump(mode="json")
                                if redact_run_value(payload) != payload:
                                    raise ValueError("Planner output contains protected values")
                        except (TypeError, ValueError):
                            state.output_attempts += 1
                            state.repair = "Planner output is invalid. Correct the output to match the `complete` schema."
                            emit(Invalid(error=state.repair, diagnostic="Invalid planner output"))
                            if state.output_attempts >= self._config.max_output_validation_attempts:
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
        self, ctx: Context[PlanningState], ev: RunSubgoal, device: DeviceResource
    ) -> SubgoalDone:
        state = await ctx.store.get_state()
        assert state.subgoal is not None
        ctx.write_event_to_stream(_stream_event(Subgoal(text=state.subgoal)))
        if private_values.get() is not None:
            ctx.write_event_to_stream(
                _stream_event(
                    Detail(section="planning", text=f"executor: {self.executor.diagnostic_name}")
                )
            )
        offset = record_offset.set(state.step)
        try:
            child = AgentWorkflow(self.executor, timeout=None).run(
                device=device,
                tools=self._tools,
                tool_config=self._tool_config,
                config=self._config,
                memory_factory=self._memory_factory,
                goal=state.subgoal,
                max_steps=min(self._config.planning.subgoal_steps, state.max_steps - state.step),
            )
        finally:
            record_offset.reset(offset)
        try:
            async for event in child.stream_events():
                match event:
                    case StopEvent():
                        # A streamed stop event would end the parent stream prematurely.
                        continue
                    case Observed(step=number):
                        event = event.model_copy(update={"step": number + state.step})
                    case StepFinished(action=action, display_action=display):
                        event = event.model_copy(
                            update={
                                "action": public_record(action).model_copy(
                                    update={"step": action.step + state.step}
                                ),
                                "display_action": (
                                    public_record(display).model_copy(
                                        update={"step": display.step + state.step}
                                    )
                                    if display
                                    else None
                                ),
                            }
                        )
                ctx.write_event_to_stream(_stream_event(event))
            result: AgentResult[Any] = await child
            final = await cast(Context[AgentState], child.ctx).store.get_state()
        finally:
            if not child.is_done():
                await child.cancel_run()
        model_result = SubgoalResult(
            goal=final.goal,
            success=result.success,
            message=(
                final.answer
                if final.reason == "complete"
                else final.answer
                or ("step limit reached" if final.reason == "max_steps" else final.reason)
            ),
            steps=result.steps,
            failure=(
                _summarize_failure(final.reason, final)
                if final.reason in _FAILURE_REASONS
                else None
            ),
        )
        return SubgoalDone(
            result=model_result,
            actions=[_offset_record(action, state.step) for action in final.actions],
            notes=final.notes,
        )

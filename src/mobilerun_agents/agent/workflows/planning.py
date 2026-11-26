"""Plan the next subgoal, then execute it in an independent agent workflow."""

import re
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Annotated, Any

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.errors import DeviceOperationNotSupported
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
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.base import Tool
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import AgentConfig, DecisionConfig, ToolsConfig

from .agent import AgentWorkflow, Begin, TransientSelectionError, prepare_goal, validate_output
from .core.result import AgentResult, FailureSummary, ScreenVisit, StepEvidence
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
    actions: list[ActionRecord]


class PlannerScreenshot(Event):
    image: bytes = Field(exclude=True, repr=False)


@dataclass
class _PlanningResources:
    device: Device
    tools: tuple[Tool, ...]
    tool_config: ToolsConfig
    config: AgentConfig
    output: type[BaseModel] | None
    memory_factory: Callable[[], Memory] | None
    model_goal: str | None = field(repr=False)
    # The checkpoint gets masked copies. Keep raw model input invocation-local.
    model_results: dict[int, SubgoalResult] = field(default_factory=dict, repr=False)
    model_remaining_plan: list[str] = field(default_factory=list, repr=False)
    model_notes: dict[str, None] = field(default_factory=dict, repr=False)
    model_subgoal: str | None = field(default=None, repr=False)


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
        render_selector_prompt(config.completion_prompt, success=True),
        render_selector_prompt(config.completion_prompt, success=False),
        render_selector_prompt(config.criteria_prompt, kind="complete_success"),
        render_selector_prompt(config.criteria_prompt, kind="complete_failure"),
    }


def _summarize_failure(
    reason: str,
    observations: Sequence[StepEvidence],
    actions: Sequence[str],
    mask: Callable[[str], str],
) -> FailureSummary:
    # Mask whole values before splitting sentences or clipping text, either of
    # which can otherwise cut a protected value in half.
    thoughts = [thought for e in observations if e.thought and (thought := mask(e.thought).strip())]

    def clip(text: str, limit: int) -> str:
        return f"{text[: limit - 1]}…" if len(text) > limit else text

    screens: dict[tuple[str, str], ScreenVisit] = {}
    previous = None
    for e in observations:
        if e.fingerprint == previous:
            continue
        previous = e.fingerprint
        if screen := screens.get(e.fingerprint):
            screen.count += 1
        else:
            screens[e.fingerprint] = ScreenVisit(
                package=mask(e.package) if e.package is not None else None,
                activity=mask(e.activity) if e.activity is not None else None,
                title=clip(mask(e.title), 100) if e.title is not None else None,
            )

    return FailureSummary(
        reason=reason,
        thoughts=tuple(clip(thought, 240) for thought in thoughts[-3:]),
        blockers=tuple(clip(blocker, 240) for blocker in _blockers(thoughts)),
        screens=tuple(screens.values()),
        actions=tuple(clip(mask(action), 180) for action in actions[-3:]),
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
        start_event, model_goal = prepare_goal(start_event, kwargs)
        token = _active_planning.set(
            _PlanningResources(
                device,
                tuple(tools),
                tool_config or ToolsConfig(),
                config,
                output,
                memory_factory,
                model_goal,
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
                    state = PlanningState(goal=redact_run_text(goal), max_steps=max_steps)
                case SubgoalDone(result=result, actions=actions):
                    state.goal = redact_run_text(state.goal)
                    state.results.append(result)
                    state.step += result.steps
                    state.actions.extend(actions)
                    state.remaining_plan = [
                        redact_run_text(goal)
                        for goal in (run.model_remaining_plan or state.remaining_plan)
                    ]
                    state.notes = list(
                        dict.fromkeys(
                            redact_run_text(note) for note in (run.model_notes or state.notes)
                        )
                    )

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
            observe_screen = bool(
                state.results
                and _needs_observation(
                    run.model_results.get(len(state.results) - 1, state.results[-1]),
                    run.config.decision,
                )
            )
            screenshot = None
            capture_screen = observe_screen and self.planner.supports_images
            while state.planner_turns < run.config.planning.max_planner_turns:
                state.planner_turns += 1
                await ctx.store.set_state(state)
                if capture_screen and screenshot is None:
                    capture_run = Run(
                        device=run.device,
                        memory=Memory.from_defaults(),
                        config=run.config,
                        macro_recorder=None,
                    )
                    try:
                        screenshot = await capture_run.screenshot()
                    except DeviceOperationNotSupported:
                        capture_screen = False
                    # Planner images are not the pre-action observations used by checkpoints.
                    if screenshot and not run_sensitive_values.get():
                        emit(PlannerScreenshot(image=screenshot))
                if debug:
                    mark = ("✓" if state.results[-1].success else "✗") if state.results else "—"
                    emit(
                        Detail(
                            section="planning",
                            text=f"turn {state.planner_turns} · remaining plan {len(state.remaining_plan)} · last result {mark}",
                        )
                    )
                request = PlanningInput(
                    goal=state.goal if run.model_goal is None else run.model_goal,
                    config=run.config,
                    remaining_plan=tuple(run.model_remaining_plan or state.remaining_plan),
                    results=tuple(
                        run.model_results.get(i, result).model_copy(deep=True)
                        for i, result in enumerate(state.results)
                    ),
                    notes=tuple(run.model_notes or state.notes),
                    package=ui.phone_state.get("packageName"),
                    activity=ui.phone_state.get("activityName"),
                    ui=ui.formatted_text,
                    output=run.output,
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
                            >= run.config.decision.max_response_repair_attempts
                        ):
                            await ctx.store.set_state(state)
                            return _stop(state, "repair_attempts")
                    case Plan(subgoals=(first, *rest)):
                        run.model_subgoal = first
                        run.model_remaining_plan[:] = rest
                        state.remaining_plan = [redact_run_text(goal) for goal in rest]
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
                        return RunSubgoal(goal=redact_run_text(first))
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
        model_goal = ev.goal if run.model_subgoal is None else run.model_subgoal
        ctx.write_event_to_stream(_stream_event(Subgoal(text=ev.goal)))
        if private_values.get() is not None:
            ctx.write_event_to_stream(
                _stream_event(
                    Detail(section="planning", text=f"executor: {self.executor.diagnostic_name}")
                )
            )
        answers: list[str] = []
        failures: list[FailureSummary] = []
        evidence: list[StepEvidence] = []
        notes: list[str] = []
        child = AgentWorkflow(self.executor, timeout=None).run(
            device=run.device,
            tools=list(run.tools),
            tool_config=run.tool_config,
            config=run.config,
            report_completion=answers.append,
            report_failure=failures.append,
            report_evidence=evidence.extend,
            report_notes=notes.extend,
            memory=run.memory_factory() if run.memory_factory else None,
            goal=model_goal,
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
                                "screenshot": (
                                    None if run_sensitive_values.get() else event.screenshot
                                ),
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
                                ),
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
                                "guard_redacted": event.guard_redacted or masked != guard,
                                "display_action": (
                                    _offset_record(display, state.step) if display else None
                                ),
                            }
                        )
                ctx.write_event_to_stream(_stream_event(event))
            result: AgentResult[Any] = await child
        finally:
            if not child.is_done():
                await child.cancel_run()
        model_failure = None
        failure = None
        if result.reason in _FAILURE_REASONS:
            recent_actions = failures[0].actions if failures else ()
            model_failure = _summarize_failure(
                result.reason, evidence, recent_actions, lambda text: text
            )
            failure = _summarize_failure(result.reason, evidence, recent_actions, redact_run_text)
        model_result = SubgoalResult(
            goal=model_goal,
            success=result.success,
            message=(
                answers[0]
                if answers
                else result.answer
                or ("step limit reached" if result.reason == "max_steps" else result.reason)
            ),
            steps=result.steps,
            failure=model_failure,
        )
        run.model_results[len(state.results)] = model_result
        if not run.model_notes:
            run.model_notes.update(dict.fromkeys(state.notes))
        run.model_notes.update(dict.fromkeys(notes))
        return SubgoalDone(
            result=model_result.model_copy(
                update={
                    "goal": redact_run_text(model_result.goal),
                    "message": redact_run_text(model_result.message),
                    "failure": failure,
                }
            ),
            actions=actions,
        )

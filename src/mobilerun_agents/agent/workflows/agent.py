"""One observer, selector and action loop; only model selection may be retried."""

import json
import re
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from hashlib import sha256
from time import monotonic
from typing import Any
from uuid import uuid4

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.macro.models import StateInput, build_screen_snapshot
from mobilerun.macro.recorder import MacroRecorder
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, ValidationError
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.handler import WorkflowHandler
from workflows.retry_policy import retry_if_exception_type, retry_policy

from mobilerun_agents.agent.action_selector.base import (
    ActionSelector,
    action_signature,
    guarded_turn,
)
from mobilerun_agents.agent.action_selector.delegating import DelegatingSelector, DelegationTurn
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall, coerce_arguments
from mobilerun_agents.agent.logs.diagnostics import (
    ActionDetail,
    Detail,
    Retry,
    Screen,
    decision_sink,
    private_values,
    record_decisions,
    redact,
    sink,
)
from mobilerun_agents.agent.logs.redaction import (
    protect_run_value,
    redact_run_arguments,
    redact_run_text,
    redact_run_value,
    run_sensitive_values,
)
from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock
from mobilerun_agents.agent.memory.blocks.run_notes import RunNotesBlock
from mobilerun_agents.agent.memory.memory import build_memory
from mobilerun_agents.agent.memory.snapshot import StepSnapshot
from mobilerun_agents.agent.run import Capture, Run
from mobilerun_agents.agent.tools.base import ActionResult, Completion, RegistryTool, Tool
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig, ToolsConfig

from .core.result import AgentResult, FailureSummary, ScreenVisit, StepEvidence
from .core.state import ActionRecord, ActionRepeat, AgentState
from .events import Act, Done, GuardDetail, Invalid, Observe, Select, StepFinished, UserMessage
from .resources import RunResource, WorkflowRun, active_run


class Begin(StartEvent):
    goal: str
    max_steps: int


def prepare_goal(
    start_event: StartEvent | None, kwargs: dict[str, Any]
) -> tuple[StartEvent | None, str | None]:
    """Keep the raw goal for model input, never in the workflow's start event."""
    match start_event:
        case Begin(goal=goal):
            return start_event.model_copy(update={"goal": redact_run_text(goal)}), goal
        case None:
            match kwargs.get("goal"):
                case str() as goal:
                    kwargs["goal"] = redact_run_text(goal)
                    return None, goal
    return start_event, None


class TransientSelectionError(ConnectionError):
    """Only an error from the model call itself may repeat selection."""


@dataclass
class _RunObservations:
    publish: Callable[[tuple[StepEvidence, ...]], None] | None
    report_notes: Callable[[tuple[str, ...]], None] | None
    model_goal: str | None = field(repr=False)
    steps: list[StepEvidence] = field(default_factory=list)
    discoveries_at_visit: dict[tuple[str, str], int] = field(default_factory=dict)
    pending_calls: dict[str, tuple[ToolCall, ...]] = field(default_factory=dict, repr=False)
    repair_errors: dict[str, str] = field(default_factory=dict, repr=False)
    delegation: DelegationTurn = field(default_factory=DelegationTurn)


_observations: ContextVar[_RunObservations] = ContextVar("agent_observations")


def _invalid(error: str, diagnostic: str = "Invalid tool arguments") -> Invalid:
    masked = redact_run_text(error)
    key = None
    if masked != error:
        key = uuid4().hex
        _observations.get().repair_errors[key] = error
    return Invalid(error=masked, diagnostic=redact_run_text(diagnostic), repair_key=key)


class _SensitiveRecorder(MacroRecorder):
    def record_action(
        self, action: Mapping[str, Any], pre_ui: StateInput = None, post_ui: StateInput = None
    ) -> dict[str, Any]:
        # A wrapper can rename or transform its sensitive arguments. Keep only
        # structural metadata, unless the tool supplied an explicit vault ref.
        safe = {"action_type": action["action_type"], "ok": action.get("ok", True)}
        if safe["action_type"] == "type_secret":
            safe.update({k: action[k] for k in ("package", "field", "name") if k in action})
        return super().record_action(safe)


def validate_output(model: type[BaseModel] | None, raw: object) -> BaseModel | None:
    return model.model_validate_json(json.dumps(raw), strict=True) if model else None


def ended(
    state: AgentState,
    reason: str,
    success: bool = False,
    output: BaseModel | None = None,
    answer: str = "",
) -> Done:
    return Done(
        result=AgentResult(
            success=success,
            reason=reason,
            answer=answer,
            output=output,
            steps=state.step,
        )
    )


def _protect_arguments(call: ToolCall, sensitive: frozenset[str]) -> None:
    for key in sensitive & call.parameters.keys():
        protect_run_value(call.parameters[key])


def _public_snapshot(snapshot: StepSnapshot) -> StepSnapshot:
    return snapshot.model_copy(
        update={
            "goal": redact_run_text(snapshot.goal),
            "notes": [redact_run_text(note) for note in snapshot.notes],
            "last_actions": [
                action.model_copy(
                    update={
                        "name": redact_run_text(action.name),
                        "args": redact_run_arguments(action.args),
                        "summary": redact_run_text(action.summary),
                    }
                )
                for action in snapshot.last_actions
            ],
            "ui_elements": [redact_run_arguments(element) for element in snapshot.ui_elements],
            "package": redact_run_text(snapshot.package) if snapshot.package else snapshot.package,
            "activity": (
                redact_run_text(snapshot.activity) if snapshot.activity else snapshot.activity
            ),
            "device_context": [
                replace(
                    item,
                    source=redact_run_text(item.source),
                    kind=redact_run_text(item.kind),
                    data=redact_run_arguments(dict(item.data)),
                )
                for item in snapshot.device_context
            ],
        }
    )


_ELEMENT_FIELDS = (
    ("text", "label", "value"),
    ("contentDescription", "content_desc", "content_description", "description"),
    ("resourceId", "resource_id", "viewIdResourceName", "id"),
    ("className", "class_name", "class", "type"),
    ("bounds", "boundsInScreen", "bounds_in_screen"),
    ("checkedState", "isChecked", "checked"),
    ("isSelected", "selected"),
)
_BLOCKER = re.compile(r"\bblocked:\s*[^\r\n]+", re.IGNORECASE)


def _element_structure(element: dict[str, JSONValue]) -> list[JSONValue]:
    match element.get("details"):
        case dict() as details:
            properties = {**details, **element}
        case _:
            properties = element
    fields = [
        next((properties[key] for key in aliases if key in properties), None)
        for aliases in _ELEMENT_FIELDS
    ]
    children: list[JSONValue] = []
    match element.get("children"):
        case list() as nodes:
            for child in nodes:
                match child:
                    case dict():
                        children.append(_element_structure(child))
    fields.append(children)
    return fields


def _fingerprint(run: Run) -> tuple[str, str]:
    assert run.snapshot is not None
    snapshot = run.snapshot
    location = sha256(json.dumps([snapshot.package, snapshot.activity]).encode()).hexdigest()
    if snapshot.ui_elements:
        structure = [_element_structure(element) for element in snapshot.ui_elements]
        screen = sha256(json.dumps(structure, sort_keys=True).encode()).hexdigest()
    elif run.capture.image is not None:
        screen = sha256(run.capture.image).hexdigest()
    else:
        screen = ""
    return location, screen


def _repeat_after(run: Run, state: AgentState, call: ToolCall) -> ActionRepeat | None:
    location, screen = _fingerprint(run)
    if not screen:
        return None
    signature = action_signature(call.name, call.parameters)
    previous = state.repeat
    assert run.snapshot is not None
    count = 1
    if previous and (previous.signature, previous.location, previous.screen) == (
        signature,
        location,
        screen,
    ):
        count = previous.count + int(run.snapshot.step > state.step)
    return ActionRepeat(
        name=call.name, signature=signature, location=location, screen=screen, count=count
    )


def _screen_title(snapshot: StepSnapshot) -> str | None:
    return next(
        (
            value
            for element in snapshot.ui_elements
            for key in ("title", "text", "contentDescription", "content_description")
            if isinstance(value := element.get(key), str) and value.strip()
        ),
        None,
    )


def _visit_screen(run: WorkflowRun, evidence: StepEvidence) -> ScreenVisit | None:
    fingerprint = evidence.fingerprint
    # Staying on the same screen is not a cycle; the stuck guard handles that case.
    if fingerprint == run.last_screen:
        return None
    run.last_screen = fingerprint
    discoveries_at_visit = _observations.get().discoveries_at_visit
    if visit := run.screens.get(fingerprint):
        visit.count = (
            visit.count + 1 if discoveries_at_visit[fingerprint] == len(run.screens) else 1
        )
    else:
        visit = run.screens[fingerprint] = ScreenVisit(
            package=evidence.package, activity=evidence.activity, title=evidence.title
        )
    discoveries_at_visit[fingerprint] = len(run.screens)
    return visit


class AgentWorkflow(Workflow):
    def __init__(self, action_selector: ActionSelector | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.action_selector = action_selector

    def run(
        self,
        ctx: Context[Any] | None = None,
        start_event: StartEvent | None = None,
        device: Device | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolsConfig | None = None,
        action_selector: ActionSelector | None = None,
        output: type[BaseModel] | None = None,
        memory: Memory | None = None,
        config: AgentConfig | None = None,
        report_completion: Callable[[str], None] | None = None,
        report_failure: Callable[[FailureSummary], None] | None = None,
        report_evidence: Callable[[tuple[StepEvidence, ...]], None] | None = None,
        report_notes: Callable[[tuple[str, ...]], None] | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        if device is None:
            raise ValueError("A device is required")
        selector = action_selector or self.action_selector
        memory = memory or build_memory(MemoryConfig(), [DeviceContextBlock()])
        memory = memory.model_copy(
            update={"memory_blocks": [*memory.memory_blocks, RunNotesBlock()]}
        )
        run = WorkflowRun(
            device=device,
            tool_config=tool_config or ToolsConfig(),
            action_selector=selector,
            pending_tools=tools if tools is not None else [],
            memory=memory,
            output=output,
            config=config or AgentConfig(),
            report_completion=report_completion,
            report_failure=report_failure,
        )
        if start_event is None:
            kwargs.setdefault("max_steps", run.config.max_steps)
        start_event, model_goal = prepare_goal(start_event, kwargs)
        token = active_run.set(run)
        observation_token = _observations.set(
            _RunObservations(report_evidence, report_notes, model_goal)
        )
        shared_values = run_sensitive_values.get()
        privacy_token = run_sensitive_values.set(
            shared_values if shared_values is not None else set()
        )
        try:
            return super().run(ctx=ctx, start_event=start_event, **kwargs)
        finally:
            run_sensitive_values.reset(privacy_token)
            _observations.reset(observation_token)
            active_run.reset(token)

    @step
    async def start(self, ctx: Context[AgentState], ev: Begin, run: RunResource) -> Observe:
        await ctx.store.set_state(AgentState(goal=redact_run_text(ev.goal), max_steps=ev.max_steps))
        return Observe()

    @step
    async def observe(
        self, ctx: Context[AgentState], ev: Observe, run: RunResource
    ) -> Select | Done:
        state = await ctx.store.get_state()
        ui = run.ui = await run.device.state.refresh()
        run.capture = Capture()
        number = state.step + 1
        observations = _observations.get()
        observations.delegation.reset_attempt()
        model_goal = observations.model_goal
        run.snapshot = StepSnapshot(
            goal=state.goal if model_goal is None else model_goal,
            notes=state.notes,
            step=number,
            last_actions=state.actions[-run.config.context_history_window :],
            package=ui.phone_state.get("packageName"),
            activity=ui.phone_state.get("activityName"),
            ui_elements=ui.elements,
            device_context=list(await run.device.context()),
        )
        observation = StepEvidence(
            step=number,
            package=run.snapshot.package,
            activity=run.snapshot.activity,
            title=_screen_title(run.snapshot),
            fingerprint=_fingerprint(run),
        )
        if observations.publish is not None:
            observations.steps.append(observation)
        if visit := _visit_screen(run, observation):
            if visit.count >= run.config.cycle_fail_after:
                message = f"cycle: reached the same screen {visit.count} times"
                if private_values.get() is not None:
                    ctx.write_event_to_stream(GuardDetail(text=message))
                return ended(state, "cycle", answer=message)
            if visit.count == run.config.cycle_note_after and state.step < state.max_steps:
                note = (
                    f"You have been on this screen {visit.count} times. Report what you found "
                    "with complete, or try a different approach."
                )
                await run.remember(ChatMessage(role="user", content=note))
                if private_values.get() is not None:
                    ctx.write_event_to_stream(GuardDetail(text=f"cycle: {note}"))
        if state.step >= state.max_steps:
            return ended(state, "max_steps")
        return Select(snapshot=_public_snapshot(run.snapshot))

    @step
    async def user_message(
        self, ctx: Context[AgentState], ev: UserMessage, run: RunResource
    ) -> None:
        await run.remember(ChatMessage(role="user", content=ev.text))

    @step
    async def repair(
        self, ctx: Context[AgentState], ev: Invalid, run: RunResource
    ) -> Select | Done:
        error = (
            _observations.get().repair_errors.pop(ev.repair_key, ev.error)
            if ev.repair_key is not None
            else ev.error
        )
        diagnostic = redact_run_text(ev.diagnostic)
        ctx.write_event_to_stream(Invalid(error=redact_run_text(ev.error), diagnostic=diagnostic))
        async with ctx.store.edit_state() as state:
            state.repair_attempts += 1
            if state.repair_attempts >= run.config.decision.max_response_repair_attempts:
                return ended(state, "repair_attempts")
            attempt = state.repair_attempts + 1
        await run.remember(
            ChatMessage(
                role="user",
                content=render_selector_prompt(
                    "assets/prompts/providers/repair.jinja2", error=error
                ),
            )
        )
        assert run.snapshot is not None
        if private_values.get() is not None:
            ctx.write_event_to_stream(Retry(attempt=attempt, reason=diagnostic))
        return Select(snapshot=_public_snapshot(run.snapshot))

    @step(retry_policy=retry_policy(retry=retry_if_exception_type(TransientSelectionError)))
    async def select(
        self, ctx: Context[AgentState], ev: Select, run: RunResource
    ) -> Act | Invalid | Done:
        turn: Run = run
        debug = private_values.get() is not None
        token = sink.set(ctx.write_event_to_stream if debug else None)
        decisions = decision_sink.set(ctx.write_event_to_stream if record_decisions.get() else None)
        guard = None
        guard_note = ""
        if debug:
            snapshot = _public_snapshot(ev.snapshot)
            ctx.write_event_to_stream(
                Screen(
                    package=snapshot.package,
                    activity=snapshot.activity,
                    elements=len(ev.snapshot.ui_elements),
                )
            )
        try:
            state = await ctx.store.get_state()
            if run.model_notes is None:
                run.model_notes = list(state.notes)
            assert run.snapshot is not None
            run.snapshot = run.snapshot.model_copy(update={"notes": list(run.model_notes)})
            run._state_snapshot = state.model_copy(
                update={
                    "goal": run.snapshot.goal,
                    "actions": run.model_actions or state.actions,
                    "notes": run.model_notes,
                },
                deep=True,
            )
            repeat = state.repeat
            if (
                repeat
                and repeat.count == run.config.stuck_exclude_after
                and state.repair_attempts == 0
            ):
                location, screen = _fingerprint(run)
                if screen and (location, screen) == (repeat.location, repeat.screen):
                    guard = repeat
                    guard_note = f"{redact_run_text(repeat.name)} repeated {repeat.count} times, screen unchanged; try something else"
                    if ctx.retry_info().retry_number == 0:
                        await run.remember(ChatMessage(role="user", content=guard_note))
                        if debug:
                            ctx.write_event_to_stream(GuardDetail(text=guard_note))
                    turn = guarded_turn(run, repeat.name, repeat.signature)
            if debug:
                ctx.write_event_to_stream(
                    Detail(section="selector", text=redact_run_text(turn.selector.diagnostic_name))
                )
            match turn.selector:
                case DelegatingSelector() as selector:
                    selection = await selector.select(turn, _observations.get().delegation)
                case selector:
                    selection = await selector.select(turn)
            # A rejected call can still identify a secret that appears in the
            # observation, diagnostics, or a later attempt.
            routed = []
            for call in selection.actions:
                member = run.registry.by_name.get(call.name)
                sensitive: frozenset[str] = frozenset()
                match member:
                    case RegistryTool(action=action) if action is not None:
                        sensitive = action.sensitive_parameters
                _protect_arguments(call, sensitive)
                routed.append((call, member, sensitive))
            if thought := selection.thought.strip():
                if steps := _observations.get().steps:
                    previous = steps[-1].thought
                    raw = f"{previous}\n{selection.thought}" if previous else selection.thought
                    steps[-1] = replace(steps[-1], thought=raw)
                run.thoughts.append(thought)
                for match in _BLOCKER.finditer(thought):
                    blocker = match.group().strip()
                    if blocker not in run.blockers:
                        run.blockers.append(blocker)
        except (TimeoutError, ConnectionError) as exc:
            raise TransientSelectionError("Model temporarily unavailable") from exc
        except InvalidResponse as exc:
            return _invalid(error=str(exc), diagnostic="Invalid model response")
        finally:
            ctx.write_event_to_stream(
                ev.model_copy(
                    update={
                        "snapshot": _public_snapshot(ev.snapshot),
                        "screenshot": None if run_sensitive_values.get() else run.capture.image,
                    }
                )
            )
            sink.reset(token)
            decision_sink.reset(decisions)
        if selection.abstained or selection.failure:
            return ended(
                await ctx.store.get_state(),
                redact_run_text(selection.failure or turn.selector.abstention_reason),
            )
        if selection.parse_error:
            return _invalid(error=selection.parse_error)
        count = len(selection.actions)
        if not count or (not turn.config.parallel_tools and count != 1):
            reason = "No tool call in reply" if not count else f"Expected one action, got {count}"
            return _invalid(error=f"{reason}; call exactly one tool", diagnostic=reason)
        for call, member, sensitive in routed:
            if call.error or call.name == "delegate" or member is None:
                return _invalid(
                    error=call.error or "Action unavailable",
                    diagnostic="Malformed action" if call.error else "Action unavailable",
                )
            try:
                schema = member.metadata.get_parameters_dict()
                # Custom selectors use the workflow's separate typed-output retry budget.
                if call.name == "complete" and "output" in schema["properties"]:
                    schema = {
                        **schema,
                        "properties": {**schema["properties"], "output": {"type": "object"}},
                    }
                call.parameters = coerce_arguments(schema, call.parameters)
                _protect_arguments(call, sensitive)
                if guard and action_signature(call.name, call.parameters) == guard.signature:
                    return _invalid(error=guard_note, diagnostic="Repeated action excluded")
                if call.name != "complete" and member.metadata.fn_schema is not None:
                    member.metadata.fn_schema.model_validate_json(
                        json.dumps(call.parameters), strict=True
                    )
            except ValidationError as exc:
                return _invalid(
                    error=str(exc.errors(include_input=False)),
                    diagnostic=", ".join(e["type"] for e in exc.errors()),
                )
            except InvalidResponse as exc:
                return _invalid(error=str(exc))
        if debug and selection.thought:
            ctx.write_event_to_stream(
                Detail(section="thought", text=redact(redact_run_text(selection.thought)))
            )
        _observations.get().delegation.accept(selection)
        if (note := selection.memory_update.strip()) and note not in run.model_notes:
            run.model_notes.append(note)
            async with ctx.store.edit_state() as state:
                state.notes = [redact_run_text(note) for note in run.model_notes]
            if debug:
                ctx.write_event_to_stream(Detail(section="memory", text=redact_run_text(note)))
        # Both the consumer event and the workflow's scheduled event are serializable.
        # Keep raw calls on the live invocation only; drop provider objects entirely.
        public = Act(
            selection=replace(
                selection,
                actions=[
                    replace(
                        call,
                        name=redact_run_text(call.name),
                        parameters=redact_run_arguments(
                            {
                                k: "[REDACTED]" if k in sensitive else redact_run_value(v)
                                for k, v in call.parameters.items()
                            }
                        ),
                        error=None,
                        call_id=None,
                    )
                    for call, _, sensitive in routed
                ],
                thought=redact_run_text(selection.thought),
                description=redact_run_text(selection.description),
                memory_update=redact_run_text(selection.memory_update),
                response=None,
                response_text=redact_run_text(selection.response_text),
                parse_result=None,
            )
        )
        key = uuid4().hex
        _observations.get().pending_calls[key] = tuple(call for call, _, _ in routed)
        ctx.write_event_to_stream(public)
        return public.model_copy(update={"dispatch_key": key})

    @step
    async def act(self, ctx: Context[AgentState], ev: Act, run: RunResource) -> Observe | Done:
        calls = (
            _observations.get().pending_calls.pop(ev.dispatch_key, None)
            if ev.dispatch_key
            else None
        )
        if calls is None:
            raise RuntimeError(
                "Pending action is unavailable; refusing to dispatch from a snapshot"
            )
        for call in calls:
            result = await self._dispatch(ctx, run, call)
            if result is not None:
                return result
        return Observe()

    async def _dispatch(
        self,
        ctx: Context[AgentState],
        run: WorkflowRun,
        call: ToolCall,
    ) -> Observe | Done | None:
        member = run.registry.by_name[call.name]
        match member:
            case RegistryTool(action=action) if action is not None:
                pass
            case _:
                raise RuntimeError("Unrouted agent tool")
        recorder = run.macro_recorder
        assert recorder is not None
        previous_recorder = recorder
        if action.sensitive_parameters:
            recorder = run.macro_recorder = _SensitiveRecorder()
        before = len(recorder.actions)
        terminal: tuple[bool, str, BaseModel | None] | None = None
        rejected = False
        started = monotonic()

        try:
            result = (await member.acall(**call.parameters)).raw_output
            match result:
                case Completion(success=success, message=message, output=output):
                    try:
                        value = validate_output(run.output, output) if success else None
                        if value is not None:
                            payload = value.model_dump(mode="json")
                            if redact_run_value(payload) != payload:
                                raise ValueError("Completion output contains protected values")
                        terminal = (success, message, value)
                        feedback = ActionResult(success, message)
                    except ValidationError as exc:
                        rejected = True
                        feedback = ActionResult(
                            False, str(exc.errors(include_input=False)), output_rejected=True
                        )
                    except ValueError:
                        rejected = True
                        feedback = ActionResult(
                            False,
                            "Completion output contains protected values",
                            output_rejected=True,
                        )
                case ActionResult() as ordinary:
                    feedback = ordinary
                case _:
                    feedback = ActionResult(True, str(result))
        except Exception as exc:
            feedback = ActionResult(False, type(exc).__name__)
            rejected = action.name == "complete"
        finally:
            run.macro_recorder = previous_recorder
        duration = monotonic() - started
        assert run.snapshot is not None
        args = redact_run_arguments(call.parameters)
        replayable = (
            "." in action.name and not action.sensitive_parameters and args == call.parameters
        )
        async with ctx.store.edit_state() as state:
            state.goal = redact_run_text(state.goal)
            if run.model_notes is not None:
                state.notes = [redact_run_text(note) for note in run.model_notes]
            repeat = _repeat_after(run, state, call) if terminal is None and not rejected else None
            state.repeat = repeat
            state.step = run.snapshot.step
            state.repair_attempts = 0
            record = ActionRecord(
                step=state.step,
                name=redact_run_text(action.name.removeprefix("device.")),
                args={
                    **args,
                    **{
                        redact_run_text(k): "[REDACTED]"
                        for k in action.sensitive_parameters & call.parameters.keys()
                    },
                },
                ok=feedback.success,
                replayable=replayable,
                summary=(
                    "[REDACTED]"
                    if action.sensitive_parameters
                    else redact_run_text(feedback.summary)
                ),
                screenshot=None,
                ui_state=None,
            )
            records = []
            for p in recorder.actions[before:]:
                raw = {
                    k: v
                    for k, v in p.items()
                    if k
                    not in {
                        "action_type",
                        "ok",
                        "pre_state",
                        "post_state",
                        "screen",
                        "recorded_at_ms",
                        "elapsed_since_previous_ms",
                    }
                }
                args = redact_run_arguments(raw)
                replayable = raw == args and (
                    not action.sensitive_parameters or p["action_type"] == "type_secret"
                )
                records.append(
                    (
                        record.model_copy(
                            update={
                                "name": redact_run_text(p["action_type"]),
                                "replayable": replayable,
                                # Base records success unless the producer marks a failure.
                                "ok": p.get("ok", True),
                                "args": args,
                            }
                        ),
                        p["pre_state"],
                    )
                )
            records = records or [(record, None)]
            state.actions.extend(executed for executed, _ in records)
            if rejected:
                state.output_attempts += 1
        if private_values.get() is not None:
            ctx.write_event_to_stream(
                ActionDetail(
                    wire=redact_run_text(call.name),
                    registry=redact_run_text(action.name),
                    args={
                        k: "***" if k in action.sensitive_parameters else v
                        for k, v in record.args.items()
                    },
                    success=record.ok,
                    summary=record.summary,
                    seconds=duration,
                )
            )
        for i, (executed, guard) in enumerate(records):
            raw_guard = guard if guard is not None else build_screen_snapshot(run.ui)
            safe_guard = redact_run_arguments(raw_guard)
            ctx.write_event_to_stream(
                StepFinished(
                    action=executed,
                    display_action=record if i == 0 else executed,
                    delegated=_observations.get().delegation.delegated,
                    pre_state=safe_guard if guard is not None or raw_guard != safe_guard else None,
                    guard_redacted=raw_guard != safe_guard,
                )
            )
        model_record = record.model_copy(
            update={
                "name": action.name.removeprefix("device."),
                "args": call.parameters,
                "summary": feedback.summary,
            }
        )
        run.model_actions.append(model_record)
        await run.remember(ChatMessage(role="assistant", content=model_record.model_dump_json()))
        if terminal is not None:
            success, message, value = terminal
            if run.report_completion is not None:
                run.report_completion(message)
            return ended(
                state, "complete", success=success, answer=redact_run_text(message), output=value
            )
        if rejected and state.output_attempts >= run.config.max_output_validation_attempts:
            return ended(state, "output_attempts")
        if repeat and repeat.count >= run.config.stuck_fail_after:
            message = f"stuck: {redact_run_text(repeat.name)} ×{repeat.count}, screen unchanged"
            if private_values.get() is not None:
                ctx.write_event_to_stream(GuardDetail(text=message))
            return ended(state, "stuck", answer=message)
        return None if feedback.success else Observe()

    @step
    async def finish(self, ctx: Context[AgentState], ev: Done, run: RunResource) -> StopEvent:
        observations = _observations.get()
        if observations.publish is not None:
            observations.publish(tuple(observations.steps))
        if observations.report_notes is not None:
            observations.report_notes(tuple(run.model_notes or ()))
        if run.report_failure is not None and ev.result.reason != "complete":
            run.report_failure(
                FailureSummary(
                    reason=ev.result.reason,
                    thoughts=tuple(run.thoughts),
                    blockers=tuple(run.blockers),
                    screens=tuple(run.screens.values()),
                    actions=tuple(
                        f"{action.name} ({'ok' if action.ok else 'failed'}): {action.summary}"
                        for action in run.model_actions[-3:]
                    ),
                )
            )
        return StopEvent(result=ev.result)

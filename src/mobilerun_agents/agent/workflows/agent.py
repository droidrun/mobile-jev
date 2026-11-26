"""One observer, selector and action loop; only model selection may be retried."""

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from time import monotonic
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.models import StateInput
from mobilerun.macro.recorder import MacroRecorder
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
from mobilerun_agents.agent.action_selector.loader import default_setup
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
from mobilerun_agents.agent.memory.memory import build_run_memory
from mobilerun_agents.agent.memory.messages import action_messages
from mobilerun_agents.agent.run import Capture, Run
from mobilerun_agents.agent.tools.base import ActionResult, Completion, RegistryTool, Tool
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig, ToolsConfig

from .core.result import AgentResult, ScreenVisit
from .core.snapshot import StepSnapshot
from .core.state import ActionOutcome, ActionRecord, ActionRepeat, AgentState
from .events import (
    Act,
    Done,
    GuardDetail,
    Invalid,
    Observe,
    Observed,
    Select,
    StepFinished,
    UserMessage,
)
from .resources import (
    DeviceResource,
    MemoryResource,
    active_device,
    active_images,
    active_memory_factory,
    record_action,
    record_offset,
    run_image_scope,
    trace_image,
)


class Begin(StartEvent):
    goal: str
    max_steps: int


class TransientSelectionError(ConnectionError):
    """Only an error from the model call itself may repeat selection."""


def _invalid(error: str, diagnostic: str = "Invalid tool arguments") -> Invalid:
    return Invalid(error=error, diagnostic=diagnostic)


class _Recorder(MacroRecorder):
    def record_action(
        self, action: Mapping[str, Any], pre_ui: StateInput = None, post_ui: StateInput = None
    ) -> dict[str, Any]:
        entry = super().record_action(action)
        entry["before_ui"] = pre_ui
        return entry


class _SensitiveRecorder(_Recorder):
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


def public_record(record: ActionOutcome) -> ActionOutcome:
    args = redact_run_arguments(record.args)
    name = redact_run_text(record.name)
    return ActionOutcome(
        step=record.step,
        name=name,
        args=args,
        ok=record.ok,
        summary=redact_run_text(record.summary),
        thought=redact_run_text(record.thought) if record.thought is not None else None,
        replayable=record.replayable and args == record.args and name == record.name,
        screen_changed=record.screen_changed,
    )


_UNCHANGED = "screen unchanged after this action (the element may be disabled or not respond)"


def _repeat_after(state: AgentState, call: ToolCall, before: tuple[str, str]) -> ActionRepeat:
    location, screen = before
    signature = action_signature(call.name, call.parameters)
    previous = state.repeat
    count = 1
    if previous and (previous.signature, previous.location, previous.screen) == (
        signature,
        location,
        screen,
    ):
        count = previous.count + 1
    return ActionRepeat(
        name=call.name, signature=signature, location=location, screen=screen, count=count
    )


def _screen_title(snapshot: StepSnapshot) -> str | None:
    return next(
        (
            value
            for element in snapshot.ui.elements
            for key in ("title", "text", "contentDescription", "content_description")
            if isinstance(value := element.get(key), str) and value.strip()
        ),
        None,
    )


def _visit_screen(state: AgentState, snapshot: StepSnapshot) -> int:
    key = ":".join(snapshot.fingerprint)
    # Staying on the same screen is not a cycle; the stuck guard handles that case.
    if key == state.last_screen:
        return 0
    state.last_screen = key
    if visit := state.screens.get(key):
        visit.count += 1
    else:
        visit = state.screens[key] = ScreenVisit(
            package=snapshot.ui.phone_state.get("packageName"),
            activity=snapshot.ui.phone_state.get("activityName"),
            title=_screen_title(snapshot),
        )
    count = (
        state.cycle_visits.get(key, 0) + 1
        if state.discoveries.get(key) == len(state.screens)
        else 1
    )
    state.discoveries[key] = len(state.screens)
    state.cycle_visits[key] = count
    return count


class AgentWorkflow(Workflow):
    _config: AgentConfig
    _tool_config: ToolsConfig
    _tools: list[Tool]
    _output: type[BaseModel] | None

    def __init__(self, action_selector: ActionSelector | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        # An injected manager belongs to the template, never an invocation.
        self._workflow_options = {
            key: value for key, value in kw.items() if key != "resource_manager"
        }
        self.action_selector = action_selector
        self._turn: Run | None = None
        self._delegation = DelegationTurn()
        self._next_snapshot: StepSnapshot | None = None
        self._image_secrets: set[str] = set()
        self._memory_lock = asyncio.Lock()

    def run(
        self,
        ctx: Context[Any] | None = None,
        start_event: StartEvent | None = None,
        device: Device | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolsConfig | None = None,
        action_selector: ActionSelector | None = None,
        output: type[BaseModel] | None = None,
        memory_factory: Callable[[], Memory] | None = None,
        config: AgentConfig | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        if device is None:
            raise ValueError("A device is required")
        # Resource caches are scoped to a Workflow instance, so use a fresh
        # invocation even when the same workflow template is run concurrently.
        workflow = type(self)(action_selector or self.action_selector, **self._workflow_options)
        workflow._config = config or AgentConfig()
        workflow._tool_config = tool_config or ToolsConfig()
        workflow._tools = tools or []
        workflow._output = output
        if start_event is None:
            kwargs.setdefault("max_steps", workflow._config.max_steps)
        device_token = active_device.set(device)
        memory_token = active_memory_factory.set(
            memory_factory or (lambda: build_run_memory(MemoryConfig(), device.platform))
        )
        shared_values = run_sensitive_values.get()
        privacy_token = run_sensitive_values.set(
            shared_values if shared_values is not None else set()
        )
        try:
            with run_image_scope() as images:
                handler = Workflow.run(workflow, ctx=ctx, start_event=start_event, **kwargs)
                if images is not None:
                    handler._result_task.add_done_callback(lambda _: images.close())
                return handler
        finally:
            run_sensitive_values.reset(privacy_token)
            active_memory_factory.reset(memory_token)
            active_device.reset(device_token)

    @step
    async def start(self, ctx: Context[AgentState], ev: Begin) -> Observe:
        await ctx.store.set_state(AgentState(goal=ev.goal, max_steps=ev.max_steps))
        return Observe()

    @step
    async def observe(
        self, ctx: Context[AgentState], ev: Observe, device: DeviceResource, memory: MemoryResource
    ) -> Select | Done:
        state = await ctx.store.get_state()
        if self._turn is None:
            if self.action_selector is None:
                setup = await default_setup()
                self.action_selector = setup.selector or setup.executor
                if self.action_selector is None:
                    raise ValueError("The default setup has no action selector.")
            run = self._turn = Run(
                device=device,
                memory=memory,
                memory_lock=self._memory_lock,
                config=self._config,
                tool_config=self._tool_config,
                output=self._output,
                images=active_images.get(),
            )
            run.registry = build_registry(run, self._tools, self._output)
        else:
            run = self._turn
        pending, self._next_snapshot = self._next_snapshot, None
        run.capture = Capture()
        number = state.step + 1
        self._delegation.reset_attempt()
        run.snapshot = pending or StepSnapshot(
            goal=state.goal,
            notes=state.notes,
            step=number,
            ui=await device.state.refresh(),
            device_context=list(await run.device.context()),
        )
        run.ui = run.snapshot.ui
        async with ctx.store.edit_state() as current:
            current.pending_thoughts.clear()
            visits = _visit_screen(current, run.snapshot)
        if visits:
            if visits >= run.config.cycle_fail_after:
                message = f"cycle: reached the same screen {visits} times"
                if private_values.get() is not None:
                    ctx.write_event_to_stream(GuardDetail(text=message))
                return ended(state, "cycle", answer=message)
            if visits == run.config.cycle_note_after and state.step < state.max_steps:
                note = (
                    f"You have been on this screen {visits} times. Report what you found "
                    "with complete, or try a different approach."
                )
                if private_values.get() is not None:
                    ctx.write_event_to_stream(GuardDetail(text=f"cycle: {note}"))
        if state.step >= state.max_steps:
            return ended(state, "max_steps")
        return Select()

    @step
    async def user_message(self, ev: UserMessage, memory: MemoryResource) -> None:
        await self._remember(memory, ChatMessage(role="user", content=ev.text))

    async def _remember(self, memory: Memory, message: ChatMessage) -> None:
        async with self._memory_lock:
            await memory.aput(message)

    @step
    async def repair(
        self, ctx: Context[AgentState], ev: Invalid, memory: MemoryResource
    ) -> Select | Done:
        run = self._turn
        assert run is not None
        diagnostic = redact_run_text(ev.diagnostic)
        ctx.write_event_to_stream(Invalid(error=redact_run_text(ev.error), diagnostic=diagnostic))
        async with ctx.store.edit_state() as state:
            state.repair_attempts += 1
            if state.repair_attempts >= run.config.decision.max_response_repair_attempts:
                return ended(state, "repair_attempts")
            attempt = state.repair_attempts + 1
        await self._remember(
            memory,
            ChatMessage(
                role="user",
                content=render_prompt(
                    "assets/prompts/providers/repair.jinja2", error=ev.error
                ),
            ),
        )
        assert run.snapshot is not None
        if private_values.get() is not None:
            ctx.write_event_to_stream(Retry(attempt=attempt, reason=diagnostic))
        return Select()

    @step(retry_policy=retry_policy(retry=retry_if_exception_type(TransientSelectionError)))
    async def select(self, ctx: Context[AgentState], ev: Select) -> Act | Invalid | Done:
        run = self._turn
        assert run is not None
        turn = run
        snapshot = run.snapshot
        assert snapshot is not None
        phone = snapshot.ui.phone_state
        observed = Observed(
            step=snapshot.step,
            package=redact_run_text(phone["packageName"]) if phone.get("packageName") else None,
            activity=redact_run_text(phone["activityName"]) if phone.get("activityName") else None,
        )
        debug = private_values.get() is not None
        token = sink.set(ctx.write_event_to_stream if debug else None)
        decisions = decision_sink.set(ctx.write_event_to_stream if record_decisions.get() else None)
        guard = None
        guard_note = ""
        if debug:
            ctx.write_event_to_stream(
                Screen(
                    package=observed.package,
                    activity=observed.activity,
                    elements=len(snapshot.ui.elements),
                )
            )
        try:
            state = await ctx.store.get_state()
            selector = self.action_selector
            assert selector is not None
            assert run.snapshot is not None
            repeat = state.repeat
            if (
                repeat
                and repeat.count == run.config.stuck_exclude_after
                and state.repair_attempts == 0
            ):
                location, screen = snapshot.fingerprint
                if screen and (location, screen) == (repeat.location, repeat.screen):
                    guard = repeat
                    guard_note = f"{repeat.name} repeated {repeat.count} times, screen unchanged; try something else"
                    if debug and ctx.retry_info().retry_number == 0:
                        ctx.write_event_to_stream(GuardDetail(text=redact_run_text(guard_note)))
                    turn = guarded_turn(run, repeat.name, repeat.signature)
            snapshot.notes = state.notes
            run._state_snapshot = turn._state_snapshot = state
            if debug:
                ctx.write_event_to_stream(
                    Detail(section="selector", text=redact_run_text(selector.diagnostic_name))
                )
            match selector:
                case DelegatingSelector():
                    selection = await selector.select(turn, self._delegation)
                case _:
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
                async with ctx.store.edit_state() as current:
                    current.pending_thoughts.append(thought)
        except (TimeoutError, ConnectionError) as exc:
            raise TransientSelectionError("Model temporarily unavailable") from exc
        except InvalidResponse as exc:
            return _invalid(error=str(exc), diagnostic="Invalid model response")
        finally:
            await self._purge_protected_images(ctx, snapshot.step)
            ctx.write_event_to_stream(
                observed.model_copy(
                    update={
                        "package": redact_run_text(observed.package) if observed.package else None,
                        "activity": (
                            redact_run_text(observed.activity) if observed.activity else None
                        ),
                    }
                )
            )
            sink.reset(token)
            decision_sink.reset(decisions)
        if selection.abstained or selection.failure:
            return ended(
                await ctx.store.get_state(),
                selection.failure or selector.abstention_reason,
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
        self._delegation.accept(selection)
        if (note := selection.memory_update.strip()) and note not in state.notes:
            async with ctx.store.edit_state() as state:
                state.notes.append(note)
            if debug:
                ctx.write_event_to_stream(Detail(section="memory", text=redact_run_text(note)))
        # The scheduled action is raw; only the separate public event is masked.
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
        ctx.write_event_to_stream(public)
        return Act(selection=replace(selection, response=None, parse_result=None))

    @step
    async def act(
        self, ctx: Context[AgentState], ev: Act, memory: MemoryResource
    ) -> Observe | Done:
        run = self._turn
        assert run is not None
        outcome: Observe | Done = Observe()
        before = run.snapshot
        assert before is not None
        try:
            for call in ev.selection.actions:
                result = await self._dispatch(ctx, run, memory, call, before)
                if self._next_snapshot is not None:
                    before = self._next_snapshot
                if result is not None:
                    outcome = result
                    break
        finally:
            assert run.snapshot is not None
            await self._purge_protected_images(ctx, run.snapshot.step)
        if not run_sensitive_values.get():
            run.save_screenshot()
            if run.capture.image and (trace := trace_image.get()) is not None:
                trace(run.capture.image)
        if (capture := record_action.get()) is not None:
            state = await ctx.store.get_state()
            for executed in state.actions:
                if executed.step == run.snapshot.step:
                    await capture(executed.model_copy(update={"step": executed.step + record_offset.get()}))
        run.capture.image = None
        async with ctx.store.edit_state() as state:
            state.pending_thoughts.clear()
        return outcome

    async def _purge_protected_images(self, ctx: Context[AgentState], step: int) -> None:
        new = (run_sensitive_values.get() or set()) - self._image_secrets
        if not new:
            return
        self._image_secrets.update(new)
        async with ctx.store.edit_state() as state:
            first = min(
                next(
                    (record.step for record in state.actions if any(
                        secret in text for text in (json.dumps(record.args, ensure_ascii=False), record.thought or "", record.summary)
                    )),
                    step - 1,
                )
                for secret in new
            )
            for record in state.actions:
                if first <= record.step <= step and (block := record.before.screenshot) is not None:
                    if block.path is not None:
                        block.path.unlink(missing_ok=True)
                    record.before.attach_screenshot(None)

    async def _dispatch(
        self,
        ctx: Context[AgentState],
        run: Run,
        memory: Memory,
        call: ToolCall,
        before: StepSnapshot,
    ) -> Observe | Done | None:
        member = run.registry.by_name[call.name]
        match member:
            case RegistryTool(action=action) if action is not None:
                pass
            case _:
                raise RuntimeError("Unrouted agent tool")
        recorder = run.macro_recorder = (
            _SensitiveRecorder() if action.sensitive_parameters else _Recorder()
        )
        fingerprint = before.fingerprint
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
            run.macro_recorder = None
        duration = monotonic() - started
        assert run.snapshot is not None
        changed = None
        if action.name != "complete":
            try:
                next_snapshot = self._next_snapshot = StepSnapshot(
                    goal=before.goal,
                    notes=before.notes,
                    step=run.snapshot.step + 1,
                    ui=await run.device.state.refresh(),
                    device_context=list(await run.device.context()),
                )
                after = next_snapshot.fingerprint
                if (
                    fingerprint[1]
                    and after[1]
                    and bool(next_snapshot.ui.elements) == bool(before.ui.elements)
                ):
                    changed = fingerprint != after
            except Exception:
                # Progress feedback is best effort; do not change the tool result.
                pass
        summary = f"{feedback.summary}\n{_UNCHANGED}" if changed is False else feedback.summary
        async with ctx.store.edit_state() as state:
            repeat = _repeat_after(state, call, fingerprint) if changed is False else None
            state.repeat = repeat
            state.no_progress = state.no_progress + 1 if changed is False else 0
            state.step = run.snapshot.step
            state.repair_attempts = 0
            record = ActionRecord(
                step=state.step,
                name=action.name.removeprefix("device."),
                args=call.parameters,
                ok=feedback.success,
                replayable="." in action.name and not action.sensitive_parameters,
                summary=summary,
                thought="\n".join(state.pending_thoughts) or None,
                screen_changed=changed,
                before=before,
            )
            records = []
            for p in recorder.actions:
                name = p["action_type"]
                if action.sensitive_parameters and name != "type_secret":
                    continue
                args = {
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
                        "before_ui",
                    }
                }
                if name == "extension_call":
                    name, args = args["name"], args["arguments"]
                primitive_ui = p.get("before_ui")
                primitive_before = before
                if (
                    isinstance(primitive_ui, UIState)
                    and primitive_ui is not before.ui
                    and not action.name.startswith("device.")
                ):
                    primitive_before = StepSnapshot(
                        at=before.at,
                        goal=before.goal,
                        step=before.step,
                        notes=before.notes,
                        ui=primitive_ui,
                        screenshot=before.screenshot,
                        device_context=before.device_context,
                    )
                records.append(
                    record.model_copy(
                        update={
                            "name": name,
                            "replayable": True,
                            "ok": p.get("ok", True),
                            "args": args,
                            "before": primitive_before,
                        }
                    )
                )
            records = records or [record]
            state.actions.extend(records)
            if rejected:
                state.output_attempts += 1
        async with self._memory_lock:
            await memory.aput_messages(action_messages(record, call.name))
        display = public_record(record)
        if action.sensitive_parameters:
            display = display.model_copy(update={"summary": "[REDACTED]"})
        if private_values.get() is not None:
            ctx.write_event_to_stream(
                ActionDetail(
                    wire=redact_run_text(call.name),
                    registry=redact_run_text(action.name),
                    args=display.args,
                    success=display.ok,
                    summary=display.summary,
                    seconds=duration,
                )
            )
        for i, executed in enumerate(records):
            ctx.write_event_to_stream(
                StepFinished(
                    action=public_record(executed),
                    display_action=display if i == 0 else None,
                    delegated=self._delegation.delegated,
                )
            )
        if terminal is not None:
            success, message, value = terminal
            return ended(state, "complete", success=success, answer=message, output=value)
        if rejected and state.output_attempts >= run.config.max_output_validation_attempts:
            return ended(state, "output_attempts")
        if repeat and repeat.count >= run.config.stuck_fail_after:
            message = f"stuck: {redact_run_text(repeat.name)} ×{repeat.count}, screen unchanged"
            if private_values.get() is not None:
                ctx.write_event_to_stream(GuardDetail(text=message))
            return ended(state, "stuck", answer=message)
        if state.no_progress >= run.config.no_progress_fail_after:
            message = f"stuck: screen unchanged after {state.no_progress} consecutive actions"
            if private_values.get() is not None:
                ctx.write_event_to_stream(GuardDetail(text=message))
            return ended(state, "stuck", answer=message)
        return None if feedback.success else Observe()

    @step
    async def finish(self, ctx: Context[AgentState], ev: Done) -> StopEvent:
        async with ctx.store.edit_state() as state:
            state.answer = ev.result.answer
            state.reason = ev.result.reason
        return StopEvent(
            result=ev.result.model_copy(
                update={
                    "answer": redact_run_text(ev.result.answer),
                    "reason": redact_run_text(ev.result.reason),
                }
            )
        )

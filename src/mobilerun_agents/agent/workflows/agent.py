"""One observer, selector and action loop; only model selection may be retried."""

import json
from collections.abc import Mapping
from dataclasses import replace
from time import monotonic
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.macro.models import StateInput
from mobilerun.macro.recorder import MacroRecorder
from pydantic import BaseModel, ValidationError
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent
from workflows.handler import WorkflowHandler
from workflows.retry_policy import retry_if_exception_type, retry_policy

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall, coerce_arguments
from mobilerun_agents.agent.logs.diagnostics import (
    ActionDetail,
    Detail,
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

from .core.result import AgentResult
from .core.state import ActionRecord, AgentState
from .events import Act, Done, Invalid, Observe, Select, StepFinished, UserMessage
from .resources import RunResource, WorkflowRun, active_run


class Begin(StartEvent):
    goal: str
    max_steps: int


class TransientSelectionError(ConnectionError):
    """Only an error from the model call itself may repeat selection."""


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
        )
        if start_event is None:
            kwargs.setdefault("max_steps", run.config.max_steps)
        token = active_run.set(run)
        shared_values = run_sensitive_values.get()
        privacy_token = run_sensitive_values.set(
            shared_values if shared_values is not None else set()
        )
        try:
            return super().run(ctx=ctx, start_event=start_event, **kwargs)
        finally:
            run_sensitive_values.reset(privacy_token)
            active_run.reset(token)

    @step
    async def start(self, ctx: Context[AgentState], ev: Begin, run: RunResource) -> Observe:
        await ctx.store.set_state(AgentState(goal=ev.goal, max_steps=ev.max_steps))
        return Observe()

    @step
    async def observe(
        self, ctx: Context[AgentState], ev: Observe, run: RunResource
    ) -> Select | Done:
        state = await ctx.store.get_state()
        if state.step >= state.max_steps:
            return ended(state, "max_steps")
        ui = run.ui = await run.device.state.refresh()
        run.capture = Capture()
        number = state.step + 1
        run.snapshot = StepSnapshot(
            goal=state.goal,
            notes=state.notes,
            step=number,
            last_actions=state.actions[-run.config.context_history_window :],
            package=ui.phone_state.get("packageName"),
            activity=ui.phone_state.get("activityName"),
            ui_elements=ui.elements,
            device_context=list(await run.device.context()),
        )
        return Select(snapshot=run.snapshot)

    @step
    async def user_message(
        self, ctx: Context[AgentState], ev: UserMessage, run: RunResource
    ) -> None:
        await run.remember(ChatMessage(role="user", content=ev.text))

    @step
    async def repair(
        self, ctx: Context[AgentState], ev: Invalid, run: RunResource
    ) -> Select | Done:
        ev = Invalid(error=redact_run_text(ev.error), diagnostic=redact_run_text(ev.diagnostic))
        ctx.write_event_to_stream(ev)
        async with ctx.store.edit_state() as state:
            state.repair_attempts += 1
            if private_values.get() is not None:
                ctx.write_event_to_stream(
                    Detail(section="repair", text=f"{state.repair_attempts}: {ev.diagnostic}")
                )
            if state.repair_attempts >= run.config.decision.max_response_repair_attempts:
                return ended(state, "repair_attempts")
        await run.remember(
            ChatMessage(
                role="user",
                content=render_selector_prompt(
                    "assets/prompts/providers/repair.jinja2", error=ev.error
                ),
            )
        )
        assert run.snapshot is not None
        return Select(snapshot=run.snapshot)

    @step(retry_policy=retry_policy(retry=retry_if_exception_type(TransientSelectionError)))
    async def select(
        self, ctx: Context[AgentState], ev: Select, run: RunResource
    ) -> Act | Invalid | Done:
        turn = run
        debug = private_values.get() is not None
        token = sink.set(ctx.write_event_to_stream if debug else None)
        decisions = decision_sink.set(ctx.write_event_to_stream if record_decisions.get() else None)
        if debug:
            ctx.write_event_to_stream(
                Screen(
                    package=ev.snapshot.package,
                    activity=ev.snapshot.activity,
                    elements=len(ev.snapshot.ui_elements),
                )
            )
        try:
            run._state_snapshot = (await ctx.store.get_state()).model_copy(deep=True)
            if debug:
                ctx.write_event_to_stream(
                    Detail(section="selector", text=turn.selector.diagnostic_name)
                )
            selection = await turn.selector.select(turn)
        except (TimeoutError, ConnectionError) as exc:
            raise TransientSelectionError("Model temporarily unavailable") from exc
        except InvalidResponse as exc:
            return Invalid(error=str(exc), diagnostic="Invalid model response")
        finally:
            ctx.write_event_to_stream(ev.model_copy(update={"screenshot": run.capture.image}))
            sink.reset(token)
            decision_sink.reset(decisions)
        # A repair can echo values from a rejected call in a later thought or note.
        routed = []
        for call in selection.actions:
            member = turn.registry.by_name.get(call.name)
            sensitive: frozenset[str] = frozenset()
            match member:
                case RegistryTool(action=action) if action is not None:
                    sensitive = action.sensitive_parameters
            _protect_arguments(call, sensitive)
            routed.append((call, member, sensitive))
        if selection.abstained or selection.failure:
            return ended(
                await ctx.store.get_state(),
                (selection.failure if selection.failure else turn.selector.abstention_reason),
            )
        if (
            selection.parse_error
            or not selection.actions
            or (not turn.config.parallel_tools and len(selection.actions) != 1)
        ):
            return Invalid(
                error=selection.parse_error or "Invalid action count",
                diagnostic=(
                    "Invalid tool arguments" if selection.parse_error else "Invalid action count"
                ),
            )
        for call, member, sensitive in routed:
            if call.error or call.name == "delegate" or member is None:
                return Invalid(
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
                if call.name != "complete" and member.metadata.fn_schema is not None:
                    member.metadata.fn_schema.model_validate_json(
                        json.dumps(call.parameters), strict=True
                    )
            except ValidationError as exc:
                return Invalid(
                    error=str(exc.errors(include_input=False)),
                    diagnostic=", ".join(e["type"] for e in exc.errors()),
                )
            except InvalidResponse as exc:
                return Invalid(error=str(exc))
        if debug and selection.thought:
            ctx.write_event_to_stream(
                Detail(section="thought", text=redact(selection.thought)[:301])
            )
        if note := redact_run_text(selection.memory_update).strip():
            async with ctx.store.edit_state() as state:
                if note not in state.notes:
                    state.notes.append(note)
                    if debug:
                        ctx.write_event_to_stream(Detail(section="memory", text=redact(note)))
        # The raw selection is an internal dispatch event, not a consumer event.
        # Raw provider objects and the parse tree can repeat the same arguments.
        ctx.write_event_to_stream(
            Act(
                selection=replace(
                    selection,
                    actions=[
                        replace(
                            call,
                            parameters={
                                k: "[REDACTED]" if k in sensitive else redact_run_value(v)
                                for k, v in call.parameters.items()
                            },
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
        )
        return Act(selection=selection)

    @step
    async def act(self, ctx: Context[AgentState], ev: Act, run: RunResource) -> Observe | Done:
        for call in ev.selection.actions:
            result = await self._dispatch(ctx, run, call)
            if result is not None:
                return result
        return Observe()

    async def _dispatch(
        self,
        ctx: Context[AgentState],
        run: Run,
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
                        message = redact_run_text(message)
                        terminal = (success, message, value)
                        feedback = ActionResult(success, message)
                    except ValidationError as exc:
                        rejected = True
                        feedback = ActionResult(
                            False, str(exc.errors(include_input=False)), output_rejected=True
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
            state.step = run.snapshot.step
            state.repair_attempts = 0
            record = ActionRecord(
                step=state.step,
                name=action.name.removeprefix("device."),
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
                                "name": p["action_type"],
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
                    wire=call.name,
                    registry=action.name,
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
            ctx.write_event_to_stream(
                StepFinished(
                    action=executed,
                    display_action=record if i == 0 else None,
                    pre_state=guard,
                )
            )
        await run.remember(ChatMessage(role="assistant", content=record.model_dump_json()))
        if terminal is not None:
            success, message, value = terminal
            return ended(state, "complete", success=success, answer=message, output=value)
        if rejected and state.output_attempts >= run.config.max_output_validation_attempts:
            return ended(state, "output_attempts")
        return None if feedback.success else Observe()

    @step
    async def finish(self, ctx: Context[AgentState], ev: Done) -> StopEvent:
        return StopEvent(result=ev.result)

"""Observe, select, execute and finish one agent invocation."""

import json
from asyncio import sleep
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.runs import recording_scope
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, PrivateAttr, ValidationError
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from mobilerun_agents.agent.action_selector.base import ActionSelector, Selection
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.events import Acted, Finished, Observed, Repaired, Selected
from mobilerun_agents.agent.tools.base import ActionResult, Completion
from mobilerun_agents.agent.tools.helpers.errors import tool_error_summary
from mobilerun_agents.config.user_config import AgentConfig

from .core.privacy import mask_value
from .core.snapshot import StepSnapshot
from .core.state import ActionRecord, AgentState, Outcome, clear_latest_action_screenshots
from .guards import check
from .resources import DeviceResource, MemoryResource, ServicesResource


class Begin(StartEvent):
    goal: str
    max_steps: int | None = None


class Observe(Event):
    pass


class Select(Event):
    thoughts: tuple[str, ...] = ()


class Act(Event):
    calls: list[ToolSelection]
    thought: str | None


class Done(Event):
    outcome: Outcome


class _VerificationPrompt(Memory):
    """One selector turn of private tool feedback; never write it to memory blocks."""

    _source: Memory = PrivateAttr()
    _feedback: list[str] = PrivateAttr()

    @classmethod
    def for_selection(cls, memory: Memory, feedback: list[str]) -> "_VerificationPrompt":
        view = cls.model_construct(**vars(memory))
        view._source, view._feedback = memory, feedback
        return view

    async def aget(
        self, input: str | ChatMessage | None = None, **block_kwargs: Any
    ) -> list[ChatMessage]:
        messages = await self._source.aget(input, **block_kwargs)
        return [*messages, ChatMessage(role="user", content="\n".join(self._feedback))]


def _step_count(state: AgentState) -> int:
    return state.actions[-1].step if state.actions else 0


def _invalid_selection(
    selection: Selection, services: ServicesResource, parallel: bool
) -> str | None:
    if not selection.calls:
        return "No tool selected; choose an available tool."
    if not parallel and len(selection.calls) != 1:
        return "Call exactly one tool."
    for call in selection.calls:
        if reason := services.registry.validate(call):
            return reason
    return None


class AgentWorkflow(Workflow):
    def __init__(
        self,
        selector: ActionSelector,
        config: AgentConfig | None = None,
        output: type[BaseModel] | None = None,
        **options: Any,
    ) -> None:
        self.selector = selector
        self.config = config or AgentConfig()
        self.output = output
        options.setdefault("timeout", self.config.workflow_timeout)
        super().__init__(**options)

    @step
    async def observe(
        self,
        ctx: Context[AgentState],
        ev: Begin | Observe,
        device: DeviceResource,
        services: ServicesResource,
    ) -> Select | Done:
        match ev:
            case Begin(goal=goal, max_steps=max_steps):
                await ctx.store.set_state(
                    AgentState(
                        goal=goal,
                        max_steps=self.config.max_steps if max_steps is None else max_steps,
                    )
                )
            case Observe() if self.config.settle_delay:
                await sleep(self.config.settle_delay)
        with recording_scope(enabled=False):
            snapshot = await StepSnapshot.observe(device)
            if not snapshot.ui.elements:
                await snapshot.capture_screenshot(device)
        async with ctx.store.edit_state() as state:
            number = _step_count(state)
            for record in reversed(state.actions):
                if record.step != number:
                    break
                record.screen_changed = record.before.compare(snapshot)
            state.current = snapshot
            state.retry_with_delegate = False
            stopped = check(state, self.config)
        if stopped is not None:
            return Done(outcome=Outcome(success=False, reason=stopped.reason))
        phone = snapshot.ui.phone_state
        ctx.write_event_to_stream(
            Observed(
                step=number + 1,
                package=phone.get("packageName"),
                activity=phone.get("activityName"),
            )
        )
        return Select()

    @step
    async def select(
        self,
        ctx: Context[AgentState],
        ev: Select,
        device: DeviceResource,
        memory: MemoryResource,
        services: ServicesResource,
    ) -> Select | Act | Done:
        registry = services.registry
        try:
            feedback = registry.take_verification_feedback()
            prompt = _VerificationPrompt.for_selection(memory, feedback) if feedback else memory
            try:
                with recording_scope(enabled=False):
                    selection = await self.selector.select(ctx, device, prompt, registry)
            finally:
                feedback.clear()
            if selection.abstained:
                abstention = selection.jev_abstention
                if abstention is None:
                    message = "Unfinished: the selector took no action."
                elif abstention.choice == "abstain":
                    message = (
                        "Unfinished: the decision model handed the subgoal back without acting "
                        "(it chose 'unfinished')."
                    )
                else:
                    if (
                        abstention.confidence is not None
                        and abstention.confidence < abstention.cutoff
                    ):
                        score = f"confidence {abstention.confidence}"
                    else:
                        score = f"target confidence {abstention.target_confidence}"
                    message = (
                        f"Unfinished: the decision model's best choice ({abstention.choice}, {score}) "
                        f"was below the cutoff {abstention.cutoff}, so no action was taken."
                    )
                return Done(
                    outcome=Outcome(
                        success=False,
                        message=message,
                        reason="abstained",
                        jev_abstention=abstention,
                    )
                )
            reason = _invalid_selection(selection, services, self.config.parallel_tools)
        except InvalidResponse as error:
            selection = Selection()
            reason = str(error)
        safe_calls, safe_thought = registry.redact_selection(selection.calls, selection.thought)
        thoughts = (*ev.thoughts, safe_thought) if safe_thought else ev.thoughts
        state = await ctx.store.get_state()
        number = _step_count(state) + 1
        if reason is None:
            ctx.write_event_to_stream(
                Selected(
                    step=number,
                    calls=safe_calls,
                    thought=safe_thought,
                    confidence=selection.confidence,
                )
            )
            return Act(calls=selection.calls, thought="\n".join(thoughts) or None)

        async with ctx.store.edit_state() as state:
            state.repairs += 1
            attempt = state.repairs
            spent = attempt > self.config.decision.max_response_repair_attempts
            state.repair = None if spent else reason
        ctx.write_event_to_stream(Repaired(step=number, attempt=attempt, reason=reason))
        if spent:
            return Done(outcome=Outcome(success=False, message=reason, reason="repair_attempts"))
        return Select(thoughts=thoughts)

    @step
    async def act(
        self,
        ctx: Context[AgentState],
        ev: Act,
        device: DeviceResource,
        services: ServicesResource,
    ) -> Observe | Done:
        state = await ctx.store.get_state()
        before = state.current
        assert before is not None
        number = _step_count(state) + 1
        registry = services.registry
        rec = device.recording_device()
        outcome = None
        output_rejections = sum(
            record.call.tool_name == "complete" and not record.ok for record in state.actions
        )
        try:
            for index, call in enumerate(ev.calls):
                (safe_call,), safe_thought = registry.redact_selection([call], ev.thought)
                resolved = registry.resolve(safe_call)
                steps: AbstractContextManager[list[int]] = (
                    rec.collect_steps() if rec is not None else nullcontext([])
                )
                step_ids: list[int] = []
                try:
                    with steps as step_ids:
                        raw = await registry.execute(call, ctx)
                    match raw:
                        case Completion():
                            outcome = self._completion(raw)
                            feedback = ActionResult(outcome.success, outcome.message or "")
                        case ActionResult():
                            feedback = raw
                        case _:
                            feedback = ActionResult(True, str(raw))
                except ValidationError as error:
                    schema = self.output if call.tool_name == "complete" else None
                    feedback = ActionResult(False, tool_error_summary(error, schema))
                except Exception as error:
                    feedback = ActionResult(False, tool_error_summary(error))
                summary = str(mask_value(feedback.summary, device.known_codes_redactor()))
                record = ActionRecord(
                    step=number,
                    call=resolved,
                    ok=feedback.success,
                    summary=summary,
                    thought=safe_thought,
                    before=before,
                )
                async with ctx.store.edit_state() as state:
                    if index == 0:
                        clear_latest_action_screenshots(state.actions)
                    state.actions.append(record)
                ctx.write_event_to_stream(Acted(action=record, step_ids=step_ids))
                if outcome is None and call.tool_name == "complete" and not record.ok:
                    output_rejections += 1
                    if output_rejections >= self.config.max_output_validation_attempts:
                        outcome = Outcome(
                            success=False, message=feedback.summary, reason="output_attempts"
                        )
                if outcome is not None:
                    break
        finally:
            async with ctx.store.edit_state() as state:
                state.repairs = 0
                state.repair = None

        return Done(outcome=outcome) if outcome is not None else Observe()

    def _completion(self, value: Completion) -> Outcome:
        output = value.output if value.success else None
        if value.success and self.output is not None:
            # Keep the validated input JSON, including any validation aliases.
            self.output.model_validate_json(json.dumps(output), strict=True)
        return Outcome(
            success=value.success,
            message=value.message,
            reason="complete",
            output=cast(JSONValue, output),
        )

    @step
    async def finish(
        self, ctx: Context[AgentState], ev: Done, services: ServicesResource
    ) -> StopEvent:
        outcome = ev.outcome
        redact_code = services.registry.device.known_codes_redactor()
        outcome = outcome.model_copy(
            update={
                "message": mask_value(outcome.message, redact_code),
                "output": mask_value(outcome.output, redact_code),
            }
        )
        async with ctx.store.edit_state() as state:
            state.outcome = outcome
        ctx.write_event_to_stream(Finished(outcome=outcome, steps=_step_count(state)))
        return StopEvent(result=ev.outcome)

"""Observe, select, execute and finish one agent invocation."""

import json
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, ImageBlock
from llama_index.core.tools import ToolSelection
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, ValidationError
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from mobilerun_agents.agent.action_selector.base import ActionSelector, Selection
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.events import Acted, Finished, Observed, Repaired, Selected
from mobilerun_agents.agent.memory.messages import remember
from mobilerun_agents.agent.tools.base import ActionResult, Completion
from mobilerun_agents.agent.tools.helpers.errors import tool_error_summary
from mobilerun_agents.config.user_config import AgentConfig

from .core.snapshot import StepSnapshot
from .core.state import ActionRecord, AgentState, Outcome
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


def _step_count(state: AgentState) -> int:
    return state.actions[-1].step if state.actions else 0


def _invalid_selection(
    selection: Selection, services: ServicesResource, parallel: bool
) -> str | None:
    if selection.abstained or not selection.calls:
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
    ) -> Select | Done:
        match ev:
            case Begin(goal=goal, max_steps=max_steps):
                await ctx.store.set_state(
                    AgentState(
                        goal=goal,
                        max_steps=self.config.max_steps if max_steps is None else max_steps,
                    )
                )
        ui = await device.state.refresh()
        snapshot = StepSnapshot(ui=ui, device_context=list(await device.context()))
        if not ui.elements:
            await snapshot.capture_screenshot(device)
        # Cache the fingerprint before an action can replace or remove the image.
        _ = snapshot.fingerprint
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
                snapshot.screenshot = None
        if stopped is not None:
            return Done(outcome=Outcome(success=False, reason=stopped.reason))
        phone = ui.phone_state
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
            selection = await self.selector.select(ctx, device, memory, registry)
            for call in selection.calls:
                registry.protect(call)
            reason = _invalid_selection(selection, services, self.config.parallel_tools)
        except InvalidResponse as error:
            selection = Selection()
            reason = str(error)
        thoughts = (*ev.thoughts, selection.thought) if selection.thought else ev.thoughts
        state = await ctx.store.get_state()
        number = _step_count(state) + 1
        if reason is None:
            ctx.write_event_to_stream(Selected(step=number, **selection.model_dump()))
            return Act(calls=selection.calls, thought="\n".join(thoughts) or None)

        async with ctx.store.edit_state() as state:
            services.images.forget(state, number, registry.protected_values)
            state.repairs += 1
            attempt = state.repairs
            spent = attempt > self.config.decision.max_response_repair_attempts
            if spent and state.current is not None:
                state.current.screenshot = None
        ctx.write_event_to_stream(Repaired(step=number, attempt=attempt, reason=reason))
        if spent:
            return Done(outcome=Outcome(success=False, message=reason, reason="repair_attempts"))
        await memory.aput_messages([ChatMessage(role="user", content=reason)])
        return Select(thoughts=thoughts)

    @step
    async def act(
        self,
        ctx: Context[AgentState],
        ev: Act,
        memory: MemoryResource,
        services: ServicesResource,
    ) -> Observe | Done:
        state = await ctx.store.get_state()
        before = state.current
        assert before is not None
        number = _step_count(state) + 1
        registry = services.registry
        records: list[ActionRecord] = []
        outcome = None
        finished = False
        output_rejections = sum(
            record.call.tool_name == "complete" and not record.ok for record in state.actions
        )
        try:
            for call in ev.calls:
                resolved = registry.resolve(call)
                memory_call = call
                try:
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
                    feedback = ActionResult(
                        False, tool_error_summary(error, schema, registry.redactor.add)
                    )
                    if schema is not None:
                        memory_call = ToolSelection(
                            tool_id=call.tool_id,
                            tool_name=call.tool_name,
                            tool_kwargs=registry.redactor.value(call.tool_kwargs),
                        )
                except Exception as error:
                    feedback = ActionResult(False, tool_error_summary(error))
                record = ActionRecord(
                    step=number,
                    call=resolved,
                    ok=feedback.success,
                    summary=feedback.summary,
                    thought=ev.thought,
                    before=before,
                )
                records.append(record)
                await remember(memory, memory_call, record)
                if outcome is None and call.tool_name == "complete" and not record.ok:
                    output_rejections += 1
                    if output_rejections >= self.config.max_output_validation_attempts:
                        outcome = Outcome(
                            success=False, message=feedback.summary, reason="output_attempts"
                        )
                if outcome is not None:
                    break
            finished = True
        finally:
            protected = registry.protected_values
            screenshot = None
            if finished and before.screenshot is not None and not protected:
                with before.screenshot.resolve_image() as image:
                    screenshot = ImageBlock(path=services.images.save(image.read()))
            persisted = before.model_copy(update={"screenshot": screenshot})
            for record in records:
                record.before = persisted
            async with ctx.store.edit_state() as state:
                state.actions.extend(records)
                services.images.forget(state, number, protected)
                state.repairs = 0
                state.current = persisted

        for record in records:
            ctx.write_event_to_stream(Acted(action=record))
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
    async def finish(self, ctx: Context[AgentState], ev: Done) -> StopEvent:
        async with ctx.store.edit_state() as state:
            state.outcome = ev.outcome
        ctx.write_event_to_stream(Finished(outcome=ev.outcome, steps=_step_count(state)))
        return StopEvent(result=ev.outcome)

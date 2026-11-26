"""Plan one subgoal, execute it, and return its evidence to the planner."""

import json
import logging
import re
from itertools import groupby
from typing import Any, cast

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, PositiveInt, ValidationError
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.events import (
    Acted,
    Detail,
    Finished,
    Observed,
    Repaired,
    Selected,
    Subgoal,
)
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory.memory import build_run_memory
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig

from .agent import AgentWorkflow
from .core.snapshot import StepSnapshot
from .core.state import ActionRecord, AgentState, Outcome, PlanningState, SubgoalResult
from .resources import DeviceResource, ServicesResource, run_resources

_log = logging.getLogger(__name__)


class Begin(StartEvent):
    goal: str
    max_steps: PositiveInt | None = None


class RunSubgoal(Event):
    goal: str


class SubgoalDone(Event):
    observe_screen: bool


def _stop(ctx: Context[PlanningState], state: PlanningState, outcome: Outcome) -> StopEvent:
    ctx.write_event_to_stream(Finished(outcome=outcome, steps=state.executor_steps))
    return StopEvent(result=outcome)


def _clip(text: str, limit: int, redact: Redactor) -> str:
    if 0 < limit < len(text):
        # Going right to left also catches earlier spans that overlap the new cut.
        for start, end in reversed(redact.masked_spans(text)):
            if start < limit < end:
                limit = start
    return text[:limit]


def _failure_summary(state: AgentState, outcome: Outcome, redact: Redactor) -> str:
    thoughts = list(dict.fromkeys(action.thought for action in state.actions if action.thought))

    def blocker_lines(thought: str) -> list[str]:
        # Splitting (or stripping) a secret can leave a fragment that won't mask later.
        if redact.text(thought) != thought:
            return [thought]
        return [line.strip() for line in re.split(r"\n|(?<=[.!?])\s+", thought)]

    blockers = dict.fromkeys(
        [
            *state.notes,
            *(action.summary for action in state.actions if not action.ok),
            *(
                line
                for thought in thoughts
                for line in blocker_lines(thought)
                if re.search(
                    r"\b(?:block\w*|can't|cannot|requires?|denied|unavailable|hides?)\b", line, re.I
                )
            ),
        ]
    )
    snapshots = [action.before for action in state.actions]
    if state.current is not None and (not snapshots or state.current.at != snapshots[-1].at):
        snapshots.append(state.current)

    def screen(snapshot: StepSnapshot) -> tuple[str, str]:
        location = snapshot.ui.phone_state
        label = f"{location.get('packageName') or 'unknown'}/{location.get('activityName') or 'unknown'}"
        text = snapshot.ui.formatted_text
        return label, _clip(text, min(120, len(text.partition("\n")[0])), redact)

    visits: dict[tuple[str, str], int] = {}
    for key, _ in groupby(snapshots, key=screen):
        visits[key] = visits.get(key, 0) + 1

    lines = [f"Failure reason: {outcome.reason}"]
    for title, items in (
        ("Observed blockers", list(blockers)[-3:]),
        ("Recent executor thoughts (observed data)", thoughts[-3:]),
        (
            "Screens visited",
            [f"{label} — {title} ×{count}" for (label, title), count in visits.items()],
        ),
        (
            "Recent action results",
            [f"{action.call.tool_name}: {action.summary}" for action in state.actions[-3:]],
        ),
    ):
        if items:
            lines.extend([f"{title}:", *(f"- {_clip(item, 240, redact)}" for item in items)])
    return "\n".join(lines)


def _offset(action: ActionRecord, offset: int) -> ActionRecord:
    return action.model_copy(update={"step": action.step + offset})


def _forget_previous_images(state: PlanningState, action_count: int) -> None:
    # The previous children did not know about the newly protected value.
    # Pixels alone cannot prove which of their observations contain it.
    def without_image(snapshot: StepSnapshot) -> StepSnapshot:
        if snapshot.screenshot is not None and snapshot.screenshot.path is not None:
            snapshot.screenshot.path.unlink(missing_ok=True)
        return snapshot.model_copy(update={"screenshot": None})

    state.actions = [
        *(
            record.model_copy(update={"before": without_image(record.before)})
            for record in state.actions[:action_count]
        ),
        *state.actions[action_count:],
    ]
    if state.current is not None:
        state.current = without_image(state.current)


class PlanningWorkflow(Workflow):
    def __init__(
        self,
        planner: Planner,
        executor: ActionSelector,
        config: AgentConfig | None = None,
        output: type[BaseModel] | None = None,
        memory_config: MemoryConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.config = config or AgentConfig()
        self.output = output
        self.output_schema = output.model_json_schema() if output else None
        self.memory_config = memory_config or MemoryConfig()
        kwargs.setdefault("timeout", self.config.workflow_timeout)
        super().__init__(**kwargs)

    @step
    async def plan(
        self, ctx: Context[PlanningState], ev: Begin | SubgoalDone, device: DeviceResource
    ) -> RunSubgoal | StopEvent:
        state = await ctx.store.get_state()
        observe_screen = False
        repair = None
        match ev:
            case Begin(goal=goal, max_steps=max_steps):
                state = PlanningState(
                    goal=goal,
                    max_steps=max_steps or self.config.planning_max_steps,
                    current=StepSnapshot(ui=await device.state.refresh()),
                )
            case SubgoalDone(observe_screen=observe_screen):
                pass
        output_attempts = 0
        while state.planner_turns < self.config.planning.max_planner_turns:
            state.planner_turns += 1
            await ctx.store.set_state(state)
            try:
                decision = await self.planner.bind(self.output_schema, repair, observe_screen).plan(
                    ctx
                )
            except InvalidResponse:
                return _stop(
                    ctx,
                    state,
                    Outcome(
                        success=False, message="Invalid planner tool call", reason="repair_attempts"
                    ),
                )
            match decision:
                case Plan(subgoals=subgoals):
                    if not subgoals or any(not goal.strip() for goal in subgoals):
                        repair = "set_plan requires a non-empty list of subgoals"
                        ctx.write_event_to_stream(Detail(section="planning", text=repair))
                        continue
                    if state.executor_steps >= state.max_steps:
                        return _stop(
                            ctx,
                            state,
                            Outcome(
                                success=False, message="Step limit reached", reason="max_steps"
                            ),
                        )
                    first, *state.plan = subgoals
                    await ctx.store.set_state(state)
                    return RunSubgoal(goal=first)
                case Finish(success=success, message=message, output=raw):
                    output: JSONValue = raw if success else None
                    if success and self.output is not None:
                        try:
                            self.output.model_validate_json(json.dumps(raw), strict=True)
                        except ValidationError:
                            output_attempts += 1
                            repair = "Planner output is invalid. Correct it to match the complete schema."
                            ctx.write_event_to_stream(Detail(section="planning", text=repair))
                            if output_attempts >= self.config.max_output_validation_attempts:
                                await ctx.store.set_state(state)
                                return _stop(
                                    ctx,
                                    state,
                                    Outcome(
                                        success=False,
                                        message=repair,
                                        reason="output_attempts",
                                    ),
                                )
                            continue
                    return _stop(
                        ctx,
                        state,
                        Outcome(success=success, message=message, reason="complete", output=output),
                    )
        await ctx.store.set_state(state)
        return _stop(
            ctx,
            state,
            Outcome(
                success=False, message="Planner turn limit reached", reason="max_planner_turns"
            ),
        )

    @step
    async def execute(
        self,
        ctx: Context[PlanningState],
        ev: RunSubgoal,
        device: DeviceResource,
        services: ServicesResource,
    ) -> SubgoalDone:
        state = await ctx.store.get_state()
        ctx.write_event_to_stream(Subgoal(text=ev.goal))
        memory = build_run_memory(self.memory_config, device.platform, self.config)
        protection = services.registry.redactor.revision
        offset = state.executor_steps
        start = len(state.actions)
        child = None
        try:
            child = AgentWorkflow(
                self.executor,
                config=self.config,
                resource_manager=await run_resources(device, memory, services),
                timeout=None,
            ).run(
                goal=ev.goal,
                max_steps=min(
                    self.config.planning.subgoal_steps, state.max_steps - state.executor_steps
                ),
            )
            child_ctx = cast(Context[AgentState], child.ctx)
            async for event in child.stream_events():
                if protection != services.registry.redactor.revision:
                    _forget_previous_images(state, start)
                    protection = services.registry.redactor.revision
                    await ctx.store.set_state(state)
                match event:
                    case Finished() | StopEvent():
                        continue
                    case Observed(step=number) | Selected(step=number) | Repaired(step=number):
                        event = event.model_copy(update={"step": number + offset})
                    case Acted(action=action):
                        child_state = await child_ctx.store.get_state()
                        state.actions = [
                            *state.actions[:start],
                            *(
                                _offset(record, offset)
                                for record in child_state.actions
                                if record.step <= action.step
                            ),
                        ]
                        await ctx.store.set_state(state)
                        event = event.model_copy(update={"action": _offset(action, offset)})
                ctx.write_event_to_stream(event)
            await child
            final = await child_ctx.store.get_state()
        finally:
            try:
                if child is not None and not child.is_done():
                    await child.cancel_run()
            finally:
                try:
                    await memory.aclose()
                except Exception:
                    _log.warning("Unable to close child run memory")

        outcome = final.outcome
        assert outcome is not None
        state.actions = [
            *state.actions[:start],
            *(_offset(record, offset) for record in final.actions),
        ]
        state.executor_steps += final.actions[-1].step if final.actions else 0
        current = final.current
        # A terminal batch skips the child's next Observe step. Keep each action's
        # before-snapshot, but do not use it as evidence for the planner's next turn.
        if current is not None and any(
            record.before.at == current.at and record.call.tool_name != "complete"
            for record in reversed(final.actions)
        ):
            current = StepSnapshot(
                ui=await device.state.refresh(), device_context=list(await device.context())
            )
            final.current = current
        state.current = current
        state.notes = list(dict.fromkeys([*state.notes, *final.notes]))
        protected = services.registry.protected_values
        redact = services.registry.redactor
        result = SubgoalResult(
            subgoal=ev.goal,
            success=outcome.success,
            message=outcome.message,
            reason=outcome.reason,
            summary="" if outcome.success else _failure_summary(final, outcome, redact),
        )
        state.results.append(result)
        ctx.write_event_to_stream(Subgoal(text=ev.goal, result=result))
        observe_screen = not result.message
        if (
            observe_screen
            and state.current is not None
            and state.current.screenshot is None
            and not protected
        ):
            try:
                image = await device.screenshot()
            except DeviceOperationNotSupported:
                pass
            else:
                if image:
                    state.current.screenshot = ImageBlock(path=services.images.save(image))
        await ctx.store.set_state(state)
        return SubgoalDone(observe_screen=observe_screen)

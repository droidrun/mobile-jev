"""Plan one subgoal, execute it, and return its evidence to the planner."""

import json
import logging
from collections.abc import Callable
from itertools import groupby
from typing import Any, Literal, cast

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, PositiveInt, ValidationError
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from mobilerun_agents.agent.action_selector import (
    ActionSelector,
    DecisionSelector,
    DelegatingSelector,
    LLMSelector,
)
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
from mobilerun_agents.agent.memory.memory import _RunMemory, prepare_child_memory
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.agent.planner.base import remaining_child_steps, remaining_steps
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig, MobileConfig

from .agent import AgentWorkflow
from .core.snapshot import StepSnapshot
from .core.state import (
    ActionRecord,
    AgentState,
    FailureSummary,
    Outcome,
    PlanningState,
    SubgoalResult,
)
from .resources import DeviceResource, MemoryResource, ServicesResource, run_resources

_log = logging.getLogger(__name__)


class Begin(StartEvent):
    goal: str
    max_steps: PositiveInt | None = None


class RunSubgoal(Event):
    goal: str
    max_steps: PositiveInt


class SubgoalDone(Event):
    pass


class PlanTurn(Event):
    pass


def _failure_summary(state: AgentState) -> FailureSummary:
    snapshots = [action.before for action in state.actions]
    if state.current is not None and (not snapshots or state.current.at != snapshots[-1].at):
        snapshots.append(state.current)

    def screen(snapshot: StepSnapshot) -> tuple[str, str]:
        location = snapshot.ui.phone_state
        label = f"{location.get('packageName') or 'unknown'}/{location.get('activityName') or 'unknown'}"
        return label, snapshot.ui.formatted_text

    visits: dict[tuple[str, str], int] = {}
    for key, _ in groupby(snapshots, key=screen):
        visits[key] = visits.get(key, 0) + 1

    return FailureSummary(
        blockers=[*state.notes, *(action.summary for action in state.actions if not action.ok)],
        thoughts=list(dict.fromkeys(action.thought for action in state.actions if action.thought)),
        screens=[(label, text, count) for (label, text), count in visits.items()],
        actions=[f"{action.call.tool_name}: {action.summary}" for action in state.actions[-3:]],
    )


def _offset(action: ActionRecord, offset: int) -> ActionRecord:
    return action.model_copy(update={"step": action.step + offset})


def _unlink_screenshot(snapshot: StepSnapshot) -> None:
    if snapshot.screenshot is not None and snapshot.screenshot.path is not None:
        snapshot.screenshot.path.unlink(missing_ok=True)


def _screen(snapshot: StepSnapshot) -> tuple[str | None, str | None]:
    phone = snapshot.ui.phone_state
    return phone.get("packageName"), phone.get("activityName")


def _record_subgoal_screen(
    result: SubgoalResult, before: StepSnapshot, after: StepSnapshot
) -> None:
    result.before_screen, result.after_screen = _screen(before), _screen(after)
    if result.before_screen != result.after_screen:
        result.screen_changed = True
    elif (changed := before.compare(after)) is not None:
        result.screen_changed = changed
    elif before.ui.formatted_text or after.ui.formatted_text:
        result.screen_changed = before.ui.formatted_text != after.ui.formatted_text


def executor_kind(
    executor: ActionSelector,
) -> Literal["decision", "decision_delegate", "llm", "custom"]:
    match executor:
        case DelegatingSelector():
            return "decision_delegate"
        case DecisionSelector():
            return "decision"
        case LLMSelector():
            return "llm"
        case _:
            return "custom"


class PlanningWorkflow(Workflow):
    def __init__(
        self,
        planner: Planner,
        executor: ActionSelector,
        config: AgentConfig | None = None,
        output: type[BaseModel] | None = None,
        memory_config: MemoryConfig | None = None,
        child_memory_factory: Callable[[], _RunMemory] | None = None,
        **kwargs: Any,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.config = config or AgentConfig()
        self.output = output
        self.output_schema = output.model_json_schema() if output else None
        self.memory_config = memory_config or MemoryConfig()
        self._child_memory_factory = child_memory_factory
        kwargs.setdefault("timeout", self.config.workflow_timeout)
        super().__init__(**kwargs)

    def _prepare_child_memory(self, platform: str) -> Callable[[], _RunMemory]:
        if self._child_memory_factory is None:
            self._child_memory_factory = prepare_child_memory(
                MobileConfig(agent=self.config, memory=self.memory_config), platform
            )
        return self._child_memory_factory

    def _stop(
        self, ctx: Context[PlanningState], state: PlanningState, outcome: Outcome
    ) -> StopEvent:
        ctx.write_event_to_stream(
            Finished(
                outcome=outcome,
                steps=state.executor_steps + state.planner_turns,
                budget=state.max_steps,
                planner_turns=state.planner_turns,
            )
        )
        return StopEvent(result=outcome)

    @step
    async def observe(
        self,
        ctx: Context[PlanningState],
        ev: Begin | SubgoalDone,
        device: DeviceResource,
        services: ServicesResource,
    ) -> PlanTurn:
        self._prepare_child_memory(device.platform)
        state = await ctx.store.get_state()
        match ev:
            case Begin(goal=goal, max_steps=max_steps):
                state = PlanningState(
                    goal=goal,
                    max_steps=max_steps or self.config.planning_max_steps,
                    executor=executor_kind(self.executor),
                )
            case SubgoalDone():
                pass
        snapshot = StepSnapshot(
            ui=await device.state.refresh(), device_context=list(await device.context())
        )
        try:
            image = await device.screenshot()
        except DeviceOperationNotSupported:
            pass
        else:
            if image:
                snapshot.screenshot = ImageBlock(path=services.images.save(image))
        # Unlike action snapshots, only the current planner observation is retained.
        if state.current is not None:
            if state.results:
                # Copy it instead of updating the result already sent on the event stream.
                result = state.results[-1].model_copy()
                _record_subgoal_screen(result, state.current, snapshot)
                state.results[-1] = result
            _unlink_screenshot(state.current)
        state.current = snapshot
        await ctx.store.set_state(state)
        phone = snapshot.ui.phone_state
        ctx.write_event_to_stream(
            Observed(
                step=state.executor_steps + state.planner_turns,
                package=phone.get("packageName"),
                activity=phone.get("activityName"),
                planner=True,
            )
        )
        return PlanTurn()

    @step
    async def plan(
        self,
        ctx: Context[PlanningState],
        ev: PlanTurn,
        memory: MemoryResource,
    ) -> RunSubgoal | StopEvent:
        state = await ctx.store.get_state()
        repair = None
        output_attempts = 0
        while remaining_steps(state) > 0:
            state.planner_turns += 1
            child_budget = min(self.config.planning.subgoal_steps, remaining_child_steps(state))
            complete_only = child_budget <= 0
            await ctx.store.set_state(state)
            try:
                decision = await self.planner.bind(self.output_schema, repair).plan(ctx, memory)
            except InvalidResponse:
                repair = "Invalid planner tool call"
                continue
            match decision:
                case Plan(subgoals=subgoals, findings=findings):
                    if complete_only:
                        repair = "Call complete when there is no budget for another subgoal."
                        continue
                    if not subgoals or any(not goal.strip() for goal in subgoals):
                        repair = "set_plan requires a non-empty list of subgoals"
                        ctx.write_event_to_stream(Detail(section="planning", text=repair))
                        continue
                    first, *state.plan = subgoals
                    for finding in findings:
                        finding = finding.strip()
                        if finding and finding not in state.notes:
                            state.notes.append(finding)
                    await ctx.store.set_state(state)
                    return RunSubgoal(goal=first, max_steps=child_budget)
                case Finish(success=False, message=message) if (
                    complete_only and not message.strip()
                ):
                    repair = "Explain what blocked the goal in the complete message."
                    continue
                case Finish(success=success, message=message, output=raw):
                    output: JSONValue = raw if success else None
                    if success and self.output is not None:
                        try:
                            self.output.model_validate_json(json.dumps(raw), strict=True)
                        except ValidationError:
                            output_attempts += 1
                            repair = "Planner output is invalid. Correct it to match the complete schema."
                            ctx.write_event_to_stream(Detail(section="planning", text=repair))
                            if (
                                not complete_only
                                and output_attempts >= self.config.max_output_validation_attempts
                            ):
                                await ctx.store.set_state(state)
                                return self._stop(
                                    ctx,
                                    state,
                                    Outcome(
                                        success=False,
                                        message=repair,
                                        reason="output_attempts",
                                    ),
                                )
                            continue
                    return self._stop(
                        ctx,
                        state,
                        Outcome(success=success, message=message, reason="complete", output=output),
                    )
        await ctx.store.set_state(state)
        return self._stop(
            ctx,
            state,
            Outcome(success=False, message="Step limit reached", reason="max_steps"),
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
        subgoal = Subgoal(
            text=ev.goal,
            step=state.executor_steps + state.planner_turns,
            budget=state.max_steps,
        )
        ctx.write_event_to_stream(subgoal)
        memory = self._prepare_child_memory(device.platform)()
        offset = state.executor_steps
        start = len(state.actions)
        child = None
        try:
            child = AgentWorkflow(
                self.executor,
                config=self.config,
                resource_manager=await run_resources(device, memory, services),
                timeout=None,
            ).run(goal=ev.goal, max_steps=ev.max_steps)
            child_ctx = cast(Context[AgentState], child.ctx)
            async for event in child.stream_events():
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
        state.notes = list(dict.fromkeys([*state.notes, *final.notes]))
        result = SubgoalResult(
            subgoal=ev.goal,
            success=outcome.success,
            message=outcome.message,
            reason=outcome.reason,
            summary=None if outcome.success else _failure_summary(final),
            jev_abstention=outcome.jev_abstention,
        )
        state.results.append(result)
        ctx.write_event_to_stream(subgoal.model_copy(update={"result": result}))
        await ctx.store.set_state(state)
        return SubgoalDone()

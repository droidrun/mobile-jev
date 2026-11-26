"""Choice-model selection, independent of transport and workflows."""

import asyncio
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Protocol

from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.action_selector.base import (
    ActionCandidate,
    ActionSelection,
    ActionSelector,
    Vision,
)
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.helpers import device_dimensions
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DecisionConfig
from mobilerun_agents.errors import SelectorCannotServe


@dataclass
class DecisionInput:
    goal: str
    state: UIState | None = None
    memory: str = ""
    history: list[Any] = field(default_factory=list)
    screenshot: bytes | None = None
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    options: tuple[ActionCandidate, ...] = ()
    instructions: str = ""

    def get_decision_prompt_variables(self, delegate_can_type: bool = False) -> dict[str, Any]:
        available = {o.action.name for o in self.options}
        can_complete = "complete" in self.registry.by_name
        if can_complete:
            available.add("complete")
        return {
            "goal": self.goal,
            "available_tools": available,
            "registered_tools": set(self.registry.by_name),
            "can_type": "input_text" in available,
            "delegate_can_type": delegate_can_type,
            "can_scroll": "swipe" in available,
            "can_locate": any(o.point_space for o in self.options),
            "can_complete": can_complete,
            "vision": self.screenshot is not None,
        }


@dataclass
class Decision:
    option_index: int | None = None
    thought: str = ""
    success: bool | None = None
    message: str = ""
    point: tuple[float, float] | None = None
    confidence: float | None = None


class DecisionModel(Protocol):
    @property
    def max_options(self) -> int | None: ...

    @property
    def locates_points(self) -> bool: ...

    async def decide(self, request: DecisionInput) -> Decision | None: ...


class DecisionSelector(ActionSelector):

    def __init__(
        self,
        model: DecisionModel,
        vision: Vision = "on",
        timeout: float | None = None,
        config: DecisionConfig | None = None,
        exclude: Collection[str] = (),
    ) -> None:
        self.config = config or DecisionConfig()
        timeout = self.config.timeout if timeout is None else timeout
        if timeout <= 0:
            raise ValueError("Decision timeout must be positive")
        self.model = model
        self.vision = vision
        self.timeout = timeout
        self.exclude = frozenset(exclude)

    async def select(self, run: Run) -> ActionSelection:
        state = run.state
        goal = state.goal
        only = self.vision == "only"
        registry = run.registry.extend((), exclude=self.exclude) if self.exclude else run.registry
        options = tuple(
            ActionCandidate(ToolCall(tool.name, tool.parameters), tool.label, context=tool.context)
            for tool in await registry.atomic(run, vision=self.vision, config=self.config)
        )
        if only and self.model.locates_points and "tap" in registry.by_name and run.ui is not None:
            provider = run.state_provider
            space = device_dimensions(run.ui, provider.requires_coordinate_tools)
            if min(space) >= 4 and (
                not provider.requires_active_contract_for_coords
                or run.ui.coordinate_contract_active
            ):
                options += (
                    ActionCandidate(
                        ToolCall("tap"),
                        render_selector_prompt(self.config.candidate_prompt, kind="tap"),
                        space,
                    ),
                )
        if not options and "complete" not in registry.by_name:
            offered = ", ".join(run.registry.by_name) or "(none)"
            raise SelectorCannotServe(f"Jev has no usable action among: {offered}.")
        screenshot = None if self.vision == "off" else await run.screenshot()
        model_request = DecisionInput(
            goal=goal,
            state=None if only else run.ui,
            screenshot=screenshot,
            history=list(state.actions[-run.config.context_history_window :]),
            memory="\n".join(m.content or "" for m in await run.history()),
            registry=registry,
            options=self._cap_options(options, self.model.max_options),
        )
        model_request.instructions = render_selector_prompt(
            self.config.system_prompt, **model_request.get_decision_prompt_variables()
        )
        decision = await asyncio.wait_for(self.model.decide(model_request), self.timeout)
        return self._to_selection(model_request, decision) or ActionSelection(abstained=True)

    def _cap_options(
        self, options: tuple[ActionCandidate, ...], limit: int | None
    ) -> tuple[ActionCandidate, ...]:
        if limit is None or len(options) <= limit:
            return options
        ranks = self.config.option_ranks
        kept = sorted(
            sorted(range(len(options)), key=lambda i: (ranks.get(options[i].action.name, 0), i))[
                :limit
            ]
        )
        return tuple(options[i] for i in kept)

    def _place(
        self, point: tuple[float, float] | None, space: tuple[float, float]
    ) -> tuple[int, int]:
        if point is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
            raise ValueError(f"Invalid located point: {point}")
        return (
            min(math.floor(point[0] * space[0]), math.ceil(space[0]) - 1),
            min(math.floor(point[1] * space[1]), math.ceil(space[1]) - 1),
        )

    def _to_selection(
        self, request: DecisionInput, decision: Decision | None
    ) -> ActionSelection | None:
        if decision is None:
            return None
        if decision.success is not None:
            if decision.option_index is not None or "complete" not in request.registry.by_name:
                raise ValueError("Completion is not available for this selection")
            action = ToolCall(
                "complete", {"success": decision.success, "message": decision.message}
            )
            description = decision.message
        else:
            index = decision.option_index
            if index is None:
                return None
            if not 0 <= index < len(request.options):
                raise ValueError(f"Invalid decision option index: {index}")
            option = request.options[index]
            action, description = option.action, option.description
            if option.point_space is not None:
                x, y = self._place(decision.point, option.point_space)
                action = ToolCall(action.name, {**action.parameters, "x": x, "y": y})
                description = render_selector_prompt(
                    self.config.candidate_prompt, kind="located", x=x, y=y
                )
        return ActionSelection(
            actions=[action],
            thought=decision.thought,
            description=description,
            confidence=decision.confidence,
        )

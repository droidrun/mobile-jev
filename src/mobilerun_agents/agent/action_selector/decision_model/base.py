"""Choice-model selection, independent of transport and workflows."""

import asyncio
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from llama_index.core.memory import Memory
from llama_index.core.tools import BaseTool, ToolSelection
from mobilerun.core.devices import Device
from mobilerun.core.state.ui_state import UIState
from workflows import Context

from mobilerun_agents.agent.action_selector.base import (
    ActionSelector,
    Selection,
    previous_action_feedback,
    screenshot_bytes,
)
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.events import Detail
from mobilerun_agents.agent.logs.memory import report_memory
from mobilerun_agents.agent.memory.messages import text_messages
from mobilerun_agents.agent.tools.atomic import AtomicTool, has_device_atomic_exception
from mobilerun_agents.agent.tools.helpers import device_dimensions, visual_tool_names
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig, Vision
from mobilerun_agents.errors import SelectorCannotServe


@dataclass
class ActionCandidate:
    action: ToolCall
    description: str
    point_space: tuple[float, float] | None = None
    context: str = ""


@dataclass
class DecisionInput:
    goal: str
    state: UIState | None = None
    memory: str = ""
    screenshot: bytes | None = None
    registry: Mapping[str, BaseTool] = field(default_factory=dict)
    options: tuple[ActionCandidate, ...] = ()
    instructions: str = ""
    intent: str = field(default="", repr=False)
    detail: Callable[[str, str], None] | None = field(default=None, repr=False)
    previous_action: str | None = None

    def get_decision_prompt_variables(self) -> dict[str, Any]:
        available = {o.action.name for o in self.options}
        return {
            "goal": self.goal,
            "available_tools": available,
            "registered_tools": set(self.registry),
            "can_type": "input_text" in available,
            "can_scroll": "swipe" in available,
            "can_locate": any(o.point_space for o in self.options),
            "can_complete": "complete" in available,
            "vision": self.screenshot is not None,
        }


@dataclass
class Decision:
    option_index: int | None = None
    thought: str = ""
    message: str | None = None
    point: tuple[float, float] | None = None
    confidence: float | None = None
    target_confidence: float | None = None


class DecisionModel(Protocol):
    @property
    def max_options(self) -> int | None: ...

    @property
    def locates_points(self) -> bool: ...

    async def decide(self, request: DecisionInput) -> Decision | None: ...


def requires_typed_completion(registry: ToolRegistry) -> bool:
    complete = registry.by_name.get("complete")
    return complete is not None and "output" in complete.metadata.get_parameters_dict().get(
        "properties", {}
    )


class DecisionSelector(ActionSelector):

    def __init__(
        self,
        model: DecisionModel,
        vision: Vision = "on",
        timeout: float | None = None,
        config: DecisionConfig | None = None,
        exclude: Collection[str] = (),
        min_confidence: float | None = None,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("Decision timeout must be positive")
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self._config = config
        self._overrides = {
            name: value
            for name, value in (("timeout", timeout), ("min_confidence", min_confidence))
            if value is not None
        }
        self.model = model
        self.vision = vision
        self.exclude = frozenset(exclude)

    @property
    def config(self) -> DecisionConfig:
        config = self._config if self._config is not None else ConfigLoader.load().agent.decision
        return config.model_copy(update=self._overrides) if self._overrides else config

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def min_confidence(self) -> float:
        return self.config.min_confidence

    async def select(
        self,
        ctx: Context[AgentState],
        device: Device,
        memory: Memory,
        registry: ToolRegistry,
        extra: tuple[AtomicTool, ...] = (),
        intent: str = "",
    ) -> Selection:
        # The wrapper advertises that it can handle typed output via its handoff option.
        if requires_typed_completion(registry) and not any(
            tool.name == "delegate" for tool in extra
        ):
            raise SelectorCannotServe(
                "Jev cannot produce typed output. Use an LLM selector or a delegate."
            )
        config = self.config
        state = await ctx.store.get_state()
        snapshot = state.current
        if snapshot is None:
            raise ValueError("Selection requires a current observation")
        ui = snapshot.ui
        only = self.vision == "only"
        tools = registry.tools()
        coordinates, indexed = visual_tool_names(tools)
        exclude = self.exclude
        if self.vision == "off":
            fixed = {
                member.metadata.get_name()
                for member in tools
                if has_device_atomic_exception(member)
            }
            exclude |= coordinates - fixed
        if only or not ui.elements:
            exclude |= indexed
        offered = {member.metadata.get_name(): member for member in registry.tools(exclude)}
        options = self._candidates(await registry.atomic_tools(exclude, ui, state.goal))
        screenshot = None if self.vision == "off" else await screenshot_bytes(ctx, snapshot, device)
        if only and screenshot and self.model.locates_points and "tap" in offered:
            provider = device.state._resolved_provider
            space = device_dimensions(ui, provider.requires_coordinate_tools)
            if min(space) >= 4 and (
                not provider.requires_active_contract_for_coords or ui.coordinate_contract_active
            ):
                options += (
                    ActionCandidate(
                        ToolCall("tap"),
                        render_prompt(config.candidate_prompt, kind="tap"),
                        space,
                    ),
                )
        limit = self.model.max_options
        if limit is not None and len(extra) > limit:
            raise ValueError("Extra options exceed the decision model option limit")
        options = self._cap_options(
            options, None if limit is None else limit - len(extra), config.option_ranks
        )
        options += self._candidates(extra)
        if not options:
            names = ", ".join(registry.by_name) or "(none)"
            raise SelectorCannotServe(f"Jev has no usable action among: {names}.")
        raw_history = await memory.aget(snapshot=snapshot, state=state)
        history = text_messages(raw_history)
        report_memory(ctx, state, memory, raw_history, history)
        model_request = DecisionInput(
            goal=state.goal,
            previous_action=previous_action_feedback(state),
            state=None if only else ui,
            screenshot=screenshot,
            memory="\n".join(m.content or "" for m in history),
            registry=offered,
            options=options,
            intent=intent,
            detail=lambda section, text: ctx.write_event_to_stream(
                Detail(section=section, text=text)
            ),
        )
        model_request.instructions = render_prompt(
            config.system_prompt,
            **model_request.get_decision_prompt_variables(),
        )
        decision = await asyncio.wait_for(self.model.decide(model_request), config.timeout)
        return self._to_selection(model_request, decision, config)

    @staticmethod
    def _candidates(atomic: tuple[AtomicTool, ...]) -> tuple[ActionCandidate, ...]:
        return tuple(
            ActionCandidate(ToolCall(tool.name, tool.parameters), tool.label, context=tool.context)
            for tool in atomic
        )

    @staticmethod
    def _cap_options(
        options: tuple[ActionCandidate, ...], limit: int | None, ranks: Mapping[str, int]
    ) -> tuple[ActionCandidate, ...]:
        completion = tuple(option for option in options if option.action.name == "complete")
        options = tuple(option for option in options if option.action.name != "complete")
        if limit is None or len(options) <= limit:
            return options + completion
        kept = sorted(
            sorted(range(len(options)), key=lambda i: (ranks.get(options[i].action.name, 0), i))[
                :limit
            ]
        )
        return tuple(options[i] for i in kept) + completion

    def _place(
        self, point: tuple[float, float] | None, space: tuple[float, float], invalid_prompt: str
    ) -> tuple[int, int]:
        if point is None or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
            raise InvalidResponse(render_prompt(invalid_prompt))
        return (
            min(math.floor(point[0] * space[0]), math.ceil(space[0]) - 1),
            min(math.floor(point[1] * space[1]), math.ceil(space[1]) - 1),
        )

    def _to_selection(
        self, request: DecisionInput, decision: Decision | None, config: DecisionConfig
    ) -> Selection:
        if decision is None:
            return Selection(abstained=True)
        min_confidence = config.min_confidence
        index = decision.option_index
        if index is None or any(
            score is not None and score < min_confidence
            for score in (decision.confidence, decision.target_confidence)
        ):
            if request.detail is not None:
                request.detail(
                    "jev abstain",
                    ("model chose abstain" if index is None else f"score below {min_confidence:g}"),
                )
            return Selection(
                abstained=True, thought=decision.thought or None, confidence=decision.confidence
            )
        if not 0 <= index < len(request.options):
            raise InvalidResponse(render_prompt(config.invalid_arguments_prompt))
        option = request.options[index]
        action = option.action
        if action.name == "complete":
            action = ToolCall(
                action.name, {**action.parameters, "message": decision.message or None}
            )
        elif option.point_space is not None:
            x, y = self._place(decision.point, option.point_space, config.invalid_arguments_prompt)
            action = ToolCall(action.name, {**action.parameters, "x": x, "y": y})
        return Selection(
            calls=[
                ToolSelection(
                    tool_id=f"call_{uuid4().hex}",
                    tool_name=action.name,
                    tool_kwargs=action.parameters,
                )
            ],
            thought=decision.thought or None,
            confidence=decision.confidence,
        )

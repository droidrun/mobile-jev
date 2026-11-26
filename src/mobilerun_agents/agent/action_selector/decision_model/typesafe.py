"""Native TypeSafe Jev choices, without an LLM compatibility layer."""

import json
import math
import traceback
from collections import Counter
from collections.abc import Callable
from time import monotonic
from typing import Any

from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import JSONValue
from mobilerun.utils.images import image_dimensions
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typesafe_sdk import AsyncTypeSafeClient, Box, Choice, Image

from mobilerun_agents.agent.action_selector.base import ActionCandidate
from mobilerun_agents.agent.auth import resolve
from mobilerun_agents.agent.logs.diagnostics import (
    DebugStep,
    DecisionOptions,
    Detail,
    decision_sink,
    protect,
    safe_text,
    sink,
)
from mobilerun_agents.agent.logs.redaction import (
    redact_run_text,
    redact_run_value,
    run_sensitive_values,
)
from mobilerun_agents.agent.tools.base import RegistryTool
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DecisionConfig

from .base import Decision, DecisionInput
from .images import prepare_image


class _Answer(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    choice: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class _Box(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    left: float = Field(ge=0, le=1, allow_inf_nan=False)
    right: float = Field(ge=0, le=1, allow_inf_nan=False)
    top: float = Field(ge=0, le=1, allow_inf_nan=False)
    bottom: float = Field(ge=0, le=1, allow_inf_nan=False)


class _Target(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    box: _Box
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class _Result(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    choices: dict[str, _Answer]
    boxes: dict[str, _Target] = Field(default_factory=dict)


def _choice_text(text: str, limit: int | None = 200) -> str:
    return safe_text(redact_run_text(text), limit)


def _choice_label(request: DecisionInput, index: int, option: ActionCandidate) -> str:
    action = option.action
    name = _choice_text(action.name, 60)
    if option.point_space is not None:
        return f"{index + 1} tap on screen"
    match request.registry.get(action.name):
        case RegistryTool(action=tool) if tool and tool.sensitive_parameters:
            return f'{index + 1} {name} "***"'
    description = _choice_text(option.description, limit=None)
    match action.parameters.get("index"):
        case int() as element_index if type(element_index) is int:
            prefix = f"{index + 1} {name} [{_choice_text(str(element_index))}]"
            # Indexed candidate descriptions can include text-field contents and state notes.
            description = description.removeprefix("Click ").removeprefix("Long press ")
            if _private_element(request.state, element_index):
                description = "***"
            else:
                description = description.partition(" (")[0]
                _, quote, quoted = description.partition('"')
                if quote and '"' in quoted:
                    description = quoted.rsplit('"', 1)[0]
        case _:
            prefix = f"{index + 1} {name}"
    label = json.dumps(safe_text(description, 100), ensure_ascii=False)
    return f"{prefix} {label}" if description else prefix


def _private_element(state: UIState | None, index: int) -> bool:
    element = state.get_element(index) if state is not None else None
    if not element:
        return False
    if element.get("isEditable") is True or element.get("isPassword") is True:
        return True
    match element.get("details"):
        case {"editable": True} | {"password": True}:
            return True
    kind = str(element.get("type") or element.get("className") or "").lower()
    return any(field in kind for field in ("edittext", "textfield", "searchfield", "searchview"))


def _log_input(
    diagnostic: Callable[[DebugStep], None],
    request: DecisionInput,
    dimensions: tuple[int, int] | None,
    locating: bool,
) -> None:
    image = f"image {dimensions[0]}×{dimensions[1]}" if dimensions is not None else "no image"
    tree = f"tree ({len(request.state.formatted_text)} chars) · " if request.state else ""
    locate = "jev locate: requested" if locating else "no locate"
    diagnostic(Detail(section="jev input", text=f"{tree}{image} · {locate}"))


def _safe_choice_value(value: JSONValue) -> JSONValue:
    match value:
        case str():
            return _choice_text(value)
        case list():
            return [_safe_choice_value(item) for item in value]
        case dict():
            return {_choice_text(key): _safe_choice_value(item) for key, item in value.items()}
        case _:
            return redact_run_value(value)


def _recorded_choices(
    request: DecisionInput, options: tuple[ActionCandidate, ...], abstain: bool
) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for i, option in enumerate(options):
        action = option.action
        sensitive: frozenset[str] = frozenset()
        match request.registry.get(action.name):
            case RegistryTool(action=tool) if tool is not None:
                sensitive = tool.sensitive_parameters
        args = {
            _choice_text(key): "***" if key in sensitive else _safe_choice_value(value)
            for key, value in action.parameters.items()
        }
        rows.append(
            {
                "label": _choice_label(request, i, option),
                "tool_call": {"name": _choice_text(action.name), "parameters": args},
            }
        )
    if abstain:
        rows.append({"label": "abstain", "tool_call": {"name": "abstain", "parameters": {}}})
    return rows


def _located_label(
    box: _Box,
    point: tuple[float, float],
    dimensions: tuple[int, int],
    space: tuple[float, float],
) -> str:
    width, height = dimensions
    image_box = f"{box.left * width:.0f},{box.top * height:.0f},{box.right * width:.0f},{box.bottom * height:.0f}"
    # Report the same clamped pixel center DecisionSelector dispatches via _place.
    x = min(math.floor(point[0] * space[0]), math.ceil(space[0]) - 1)
    y = min(math.floor(point[1] * space[1]), math.ceil(space[1]) - 1)
    return f"tap on screen · box ({image_box}) in image px → tap ({x},{y}) device px"


class TypeSafeDecisionModel:
    locates_points = True

    def __init__(
        self,
        model: str | None = None,
        min_confidence: float | None = None,
        config: DecisionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or DecisionConfig()
        min_confidence = self.config.min_confidence if min_confidence is None else min_confidence
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self._client_kwargs = dict(kwargs)
        if model is not None:
            self._client_kwargs["model"] = model
        self.min_confidence = min_confidence

    @property
    def max_options(self) -> int:
        return self.config.max_options

    async def decide(self, request: DecisionInput) -> Decision | None:
        if not request.options:
            raise ValueError("TypeSafe decisions require non-empty action options")
        indices = {}
        for i, candidate in enumerate(request.options):
            action = candidate.action
            label = f"action_{i}"
            if action.name == "complete":
                label = "complete_success" if action.parameters["success"] else "complete_failure"
            indices[label] = i
        criteria: dict[str, Any] = {
            label: {
                "description": request.options[i].description,
                "tool": request.options[i].action.name,
                "parameters": request.options[i].action.parameters,
                **({"context": request.options[i].context} if request.options[i].context else {}),
            }
            for label, i in indices.items()
        }
        abstain = not any(option.action.name == "delegate" for option in request.options)
        if abstain:
            criteria["abstain"] = render_selector_prompt(
                self.config.criteria_prompt, kind="abstain"
            )
        if len(criteria) > self.config.max_choices:
            raise ValueError(
                f"TypeSafe accepts at most {self.config.max_choices} choices, got {len(criteria)}"
            )
        state: dict[str, Any] = {
            "goal": request.goal,
            "memory": request.memory,
            "history": TypeAdapter(list[Any]).dump_python(request.history, mode="json"),
        }
        if request.intent:
            state["intent"] = request.intent
        locating = request.screenshot is not None and any(o.point_space for o in request.options)
        if request.state is not None:
            state.update(
                ui=request.state.formatted_text,
                focused_text=request.state.focused_text,
                phone_state=request.state.phone_state,
            )
        questions: dict[str, Any] = {}
        request_kwargs: dict[str, Any] = {}
        coverage = (1.0, 1.0)
        diagnostic = sink.get()
        dimensions = None
        if request.screenshot is not None:
            prepared, coverage = prepare_image(request.screenshot, self.config)
            if diagnostic:
                dimensions = image_dimensions(prepared)
            state["screenshot"] = Image(prepared)
            request_kwargs["extra_headers"] = {"TypeSafe-Danger-Zone-Images": "true"}
            if locating:
                questions["target"] = Box(
                    instructions=render_selector_prompt(
                        self.config.locate_prompt, kind="instructions"
                    )
                )
        instructions = request.instructions or render_selector_prompt(
            self.config.system_prompt, **request.get_decision_prompt_variables()
        )
        options = dict(self._client_kwargs)
        api_key = (await resolve("typesafe", options.get("api_key"))).api_key
        options["api_key"] = api_key
        protect(api_key)
        save = decision_sink.get()
        if diagnostic:
            _log_input(diagnostic, request, dimensions, locating)
        if diagnostic or save:
            # Include the provider key without adding it to the run's tracked device inputs.
            scope = run_sensitive_values.set((run_sensitive_values.get() or set()) | {api_key})
            try:
                groups = Counter(_choice_text(option.action.name) for option in request.options)
                grouped = ", ".join(f"{name} {count}" for name, count in groups.most_common())
                summary = f"{len(request.options)} · {grouped}"
                if abstain:
                    summary += " · +abstain"
                if save:
                    choices = _recorded_choices(request, request.options, abstain)
                    save(DecisionOptions(summary=summary, choices=choices))
                elif diagnostic:
                    diagnostic(Detail(section="jev choices", text=summary))
            finally:
                run_sensitive_values.reset(scope)
        started = monotonic() if diagnostic else 0.0
        try:
            async with AsyncTypeSafeClient(**options) as client:
                raw = await client.system_one(
                    state=state,
                    questions={
                        "action": Choice(instructions=instructions, criteria=criteria),
                        **questions,
                    },
                    **request_kwargs,
                )
            result = _Result.model_validate(raw)
        except Exception as exc:
            if diagnostic:
                diagnostic(
                    Detail(
                        section="jev answer",
                        text=f"unavailable · latency={monotonic() - started:.3f}s",
                    )
                )
            frames = " -> ".join(
                f"{frame.f_code.co_filename}:{line} in {frame.f_code.co_name}"
                for frame, line in traceback.walk_tb(exc.__traceback__)
            )
            message = f"TypeSafe request failed ({type(exc).__name__})"
            error: Exception
            match exc:
                case TimeoutError():
                    error = TimeoutError(message)
                case ConnectionError():
                    error = ConnectionError(message)
                case _:
                    error = RuntimeError(message)
            error.add_note(frames)
            raise error from None
        answer = result.choices.get("action")
        if answer is None or answer.choice not in criteria:
            raise ValueError("TypeSafe returned an unknown or missing action choice")
        latency = monotonic() - started if diagnostic else 0.0
        confidence = answer.confidence
        confidence_text = f"confidence={confidence:g}"
        if self.min_confidence > 0:
            confidence_text += f" (min={self.min_confidence:g})"
        index = indices.get(answer.choice)
        option = request.options[index] if index is not None else None
        space = option.point_space if option is not None else None
        point = None
        target = None
        reason = None
        if confidence < self.min_confidence:
            reason = f"confidence {confidence:g} < {self.min_confidence:g}"
        elif answer.choice == "abstain":
            reason = "model chose abstain"
        elif space is not None:
            target = result.boxes.get("target")
            if target is None:
                if diagnostic:
                    diagnostic(
                        Detail(
                            section="jev answer",
                            text=f"tap on screen · missing target box · {confidence_text} · latency={latency:.3f}s",
                        )
                    )
                raise ValueError("TypeSafe did not locate the on-screen tap")
            if target.confidence is not None and target.confidence < self.min_confidence:
                reason = f"target confidence {target.confidence:g} < {self.min_confidence:g}"
            else:
                box = target.box
                point = (
                    min(1.0, (box.left + box.right) / 2 / coverage[0]),
                    min(1.0, (box.top + box.bottom) / 2 / coverage[1]),
                )
        if diagnostic:
            label = (
                _choice_label(request, index, option)
                if index is not None and option is not None
                else answer.choice
            )
            locate = " · locate unused" if locating and space is None else ""
            if target is not None:
                if point is not None and dimensions is not None and space is not None:
                    label = _located_label(target.box, point, dimensions, space)
                if target.confidence is not None:
                    confidence_text += f" · target confidence={target.confidence:g}"
            diagnostic(
                Detail(
                    section="jev answer",
                    text=f"{label}{locate} · {confidence_text} · latency={latency:.3f}s",
                )
            )
            if reason is not None:
                diagnostic(Detail(section="jev abstain", text=reason))
        if reason is not None:
            return None
        if option is not None and option.action.name == "complete":
            return Decision(
                option_index=index,
                confidence=confidence,
                message=render_selector_prompt(
                    self.config.completion_prompt, success=option.action.parameters["success"]
                ),
            )
        return Decision(option_index=index, confidence=confidence, point=point)


Jev = TypeSafeDecisionModel

"""Native TypeSafe Jev choices, without an LLM compatibility layer."""

import traceback
from collections import Counter
from time import monotonic
from typing import Any

from mobilerun.utils.images import image_dimensions
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typesafe_sdk import AsyncTypeSafeClient, Box, Choice, Image

from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.auth import resolve
from mobilerun_agents.config.prompts import render_prompt
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


class TypeSafeDecisionModel:
    locates_points = True

    def __init__(
        self,
        model: str | None = None,
        config: DecisionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or DecisionConfig()
        self._client_kwargs = dict(kwargs)
        if model is not None:
            self._client_kwargs["model"] = model

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
        criteria["abstain"] = render_prompt(self.config.criteria_prompt, kind="abstain")
        if len(criteria) > self.config.max_choices:
            raise ValueError(
                f"TypeSafe accepts at most {self.config.max_choices} choices, got {len(criteria)}"
            )
        state: dict[str, Any] = {
            "goal": request.goal,
            "memory": request.memory,
        }
        if request.previous_action is not None:
            state["previous_action"] = request.previous_action
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
        diagnostic = request.detail
        dimensions = None
        if request.screenshot is not None:
            prepared, coverage = prepare_image(request.screenshot, self.config)
            if diagnostic:
                dimensions = image_dimensions(prepared)
            state["screenshot"] = Image(prepared)
            request_kwargs["extra_headers"] = {"TypeSafe-Danger-Zone-Images": "true"}
            if locating:
                questions["target"] = Box(
                    instructions=render_prompt(self.config.locate_prompt, kind="instructions")
                )
        instructions = request.instructions or render_prompt(
            self.config.system_prompt, **request.get_decision_prompt_variables()
        )
        options = dict(self._client_kwargs)
        api_key = (await resolve("typesafe", options.get("api_key"))).api_key
        options["api_key"] = api_key
        if diagnostic:
            image = f"image {dimensions[0]}×{dimensions[1]}" if dimensions else "no image"
            tree = f"tree ({len(request.state.formatted_text)} chars) · " if request.state else ""
            diagnostic(
                "jev input", f"{tree}{image} · {'locate requested' if locating else 'no locate'}"
            )
            groups = Counter(option.action.name for option in request.options)
            grouped = ", ".join(f"{name} {count}" for name, count in groups.most_common())
            diagnostic(
                "jev choices",
                f"{len(request.options)} · {grouped} · +abstain",
            )
        started = monotonic() if diagnostic else 0.0
        validating_result = False
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
            validating_result = True
            result = _Result.model_validate(raw)
        except Exception as exc:
            if diagnostic:
                diagnostic("jev answer", f"unavailable · latency={monotonic() - started:.3f}s")
            frames = " -> ".join(
                f"{frame.f_code.co_filename}:{line} in {frame.f_code.co_name}"
                for frame, line in traceback.walk_tb(exc.__traceback__)
            )
            message = f"TypeSafe request failed ({type(exc).__name__})"
            error: Exception
            match exc:
                case ValidationError() if validating_result and all(
                    len(issue["loc"]) == 4 and issue["loc"][:3] == ("boxes", "target", "box")
                    for issue in exc.errors(include_input=False, include_context=False)
                ):
                    raise InvalidResponse(
                        render_prompt(self.config.invalid_arguments_prompt)
                    ) from None
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
            raise InvalidResponse(render_prompt(self.config.invalid_arguments_prompt))
        latency = monotonic() - started if diagnostic else 0.0
        confidence = answer.confidence
        confidence_text = f"confidence={confidence:g}"
        index = indices.get(answer.choice)
        option = request.options[index] if index is not None else None
        space = option.point_space if option is not None else None
        point = None
        target = None
        if space is not None:
            target = result.boxes.get("target")
            if target is None:
                if diagnostic:
                    diagnostic(
                        "jev answer",
                        f"tap on screen · missing target box · {confidence_text} · latency={latency:.3f}s",
                    )
            else:
                box = target.box
                point = (
                    min(1.0, (box.left + box.right) / 2 / coverage[0]),
                    min(1.0, (box.top + box.bottom) / 2 / coverage[1]),
                )
        if diagnostic:
            label = option.action.name if option is not None else answer.choice
            locate = " · locate unused" if locating and space is None else ""
            if target is not None:
                if target.confidence is not None:
                    confidence_text += f" · target confidence={target.confidence:g}"
            diagnostic(
                "jev answer", f"{label}{locate} · {confidence_text} · latency={latency:.3f}s"
            )
        return Decision(
            option_index=index,
            confidence=confidence,
            point=point,
            target_confidence=target.confidence if target is not None else None,
        )

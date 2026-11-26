"""Agent settings, independent of the base device configuration."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from llama_index.core.memory import InsertMethod
from llama_index.core.memory.memory import DEFAULT_TOKEN_LIMIT
from mobilerun.config.user_config import TelemetryConfig, TracingConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from mobilerun_agents.config.settings import (
    DEFAULT_FAST_SYSTEM_PROMPT,
    DEFAULT_MAX_STEPS,
)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


Vision = Literal["off", "on", "only"]
ScrollDirection = Literal["up", "down", "left", "right"]
Fraction = Annotated[float, Field(ge=0, le=1)]
SwipePath = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


class LLMProfile(Section):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    vision: Vision | None = None
    temperature: float | None = None
    base_url: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DecisionProfile(Section):
    jev_only: bool = False
    vision: Vision = "on"
    provider: Literal["typesafe"] = "typesafe"
    model: str | None = None
    base_url: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DecisionConfig(Section):
    max_response_repair_attempts: PositiveInt = 3
    system_prompt: str = "assets/prompts/decision/system.jinja2"
    timeout: PositiveFloat = 120
    min_confidence: Annotated[float, Field(ge=0, le=1)] = 0
    max_options: Annotated[int, Field(gt=0, le=252)] = 252
    max_choices: Annotated[int, Field(gt=0, le=255)] = 255
    max_schema_combinations: PositiveInt = 18
    option_ranks: dict[str, int] = {"tap_element": 1, "tap": 1, "start_app": 2}
    image_min_side: PositiveInt = 280
    image_max_side: Annotated[int, Field(gt=0, le=1568)] = 1568
    jpeg_quality: Annotated[int, Field(ge=1, le=100)] = 85
    candidate_prompt: str = "assets/prompts/decision/candidate.jinja2"
    criteria_prompt: str = "assets/prompts/decision/criteria.jinja2"
    locate_prompt: str = "assets/prompts/decision/locate.jinja2"
    completion_prompt: str = "assets/prompts/decision/completion.jinja2"
    borrowed_text_limit: PositiveInt = 3
    edge_share: Annotated[float, Field(gt=0, lt=0.5)] = 0.1
    max_owner_screen_share: Annotated[float, Field(gt=0, le=1)] = 0.5
    knob_slack: Annotated[int, Field(ge=0)] = 4
    llm_request_prompt: str = "assets/prompts/llm/selection_request.jinja2"
    invalid_arguments_prompt: str = "assets/prompts/llm/invalid_arguments.jinja2"


class DelegationConfig(Section):
    primary: str | None = None
    delegate: str = "delegate"
    min_confidence: Annotated[float | None, Field(ge=0, le=1)] = 0.7
    invalid_prompt: str = "assets/prompts/delegate/invalid.jinja2"


class PlanningConfig(Section):
    subgoal_steps: PositiveInt = 10
    max_planner_turns: PositiveInt = 10


class AgentConfig(Section):
    selector: str = "default"
    max_steps: PositiveInt = DEFAULT_MAX_STEPS
    planning_max_steps: PositiveInt = 40
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    workflow_timeout: PositiveFloat | None = None
    max_output_validation_attempts: PositiveInt = 2
    context_history_window: PositiveInt = 5

    system_prompt: str = DEFAULT_FAST_SYSTEM_PROMPT
    parallel_tools: bool = True
    streaming: bool = True
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    delegation: DelegationConfig | None = None


class ToolsConfig(Section):
    exclude: list[str] | None = None
    app_open_settle_delay: Annotated[float, Field(ge=0)] = 1
    wait_duration: PositiveFloat = 1
    swipe_duration: PositiveFloat = 0.5
    swipe_paths: Annotated[dict[ScrollDirection, SwipePath], Field(min_length=4)] = {
        "up": ((0.5, 0.2), (0.5, 0.8)),
        "down": ((0.5, 0.8), (0.5, 0.2)),
        "left": ((0.2, 0.5), (0.8, 0.5)),
        "right": ((0.8, 0.5), (0.2, 0.5)),
    }


class CheckpointsConfig(Section):
    enabled: bool = True
    folder: Path | None = None


class MemoryConfig(Section):
    token_limit: PositiveInt = DEFAULT_TOKEN_LIMIT
    insert_method: InsertMethod = InsertMethod.USER


class MobileConfig(Section):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    default_llm: str | None = None  # Profile name. Jev may pin a different delegate.
    llm_profiles: dict[str, LLMProfile] = Field(default_factory=dict)
    decision_profiles: dict[str, DecisionProfile] = Field(default_factory=dict)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    analytics: TelemetryConfig = Field(default_factory=TelemetryConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))  # type: ignore[no-untyped-call]

"""Agent settings, independent of the base device configuration."""

import re
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

import yaml
from llama_index.core.memory import InsertMethod
from llama_index.core.memory.memory import DEFAULT_TOKEN_LIMIT
from mobilerun.config.user_config import TelemetryConfig, TracingConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from mobilerun_agents.config.settings import (
    DEFAULT_FAST_SYSTEM_PROMPT,
    DEFAULT_MAX_STEPS,
)


class Section(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False, hide_input_in_errors=True
    )


Vision = Literal["off", "on", "only"]
Reasoning = Literal["low", "medium", "high"]
ProviderKind = Literal["llm", "decision"]
PROVIDER_KINDS: dict[str, ProviderKind] = {
    "anthropic": "llm",
    "deepseek": "llm",
    "gemini": "llm",
    "grok": "llm",
    "minimax": "llm",
    "ollama": "llm",
    "openai": "llm",
    "openai_like": "llm",
    "openrouter": "llm",
    "typesafe": "decision",
    "zai": "llm",
}
ScrollDirection = Literal["up", "down", "left", "right"]
Fraction = Annotated[float, Field(ge=0, le=1)]
SwipePath = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


CREDENTIAL_MESSAGE = "Profiles cannot contain credentials; use `mobilerun agent providers add`."
_BASE_URL_MESSAGE = (
    "Profile base URL must not contain userinfo, a query string, or a fragment; "
    "use `mobilerun agent providers add` for credentials."
)


def _credential_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized in {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "credentials",
        "headers",
        "kwargs",
        "pass",
        "pwd",
        "sig",
    } or normalized.endswith(
        ("key", "token", "password", "passwd", "secret", "credential", "signature")
    )


class _Profile(Section):
    provider: str = Field(min_length=1)
    _expected_kind: ClassVar[ProviderKind]

    @property
    def kind(self) -> ProviderKind:
        return self._expected_kind

    @field_validator("provider")
    @classmethod
    def _provider_kind(cls, provider: str) -> str:
        if (kind := PROVIDER_KINDS.get(provider)) is None:
            raise PydanticCustomError(
                "profile_provider", "Unknown provider; see `mobilerun agent profiles add --help`."
            )
        if kind != cls._expected_kind:
            raise ValueError(f"Provider does not support {cls._expected_kind} profile settings")
        return provider

    @model_validator(mode="before")
    @classmethod
    def _reject_credential_fields(cls, value: object) -> object:
        if type(value) is dict and any(_credential_name(str(key)) for key in value):
            raise PydanticCustomError("profile_credentials", CREDENTIAL_MESSAGE)
        return value


class LLMProfile(_Profile):
    _expected_kind = "llm"
    model: str = Field(min_length=1)
    reasoning: Reasoning | None = None
    vision: Vision | None = None
    temperature: float | None = None
    base_url: str | None = None

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        url = HttpUrl(value)
        # HttpUrl strips an empty userinfo marker (`https://@host`). Check the
        # original authority as well as the parsed URL.
        authority = value.partition(":")[2].replace("\\", "/").lstrip("/").partition("/")[0]
        if (
            "@" in authority
            or url.username is not None
            or url.password is not None
            or url.query is not None
            or url.fragment is not None
        ):
            raise PydanticCustomError("profile_credentials", _BASE_URL_MESSAGE)
        return str(url)


class DecisionProfile(_Profile):
    _expected_kind = "decision"
    vision: Vision | None = None
    min_confidence: Fraction | None = None


Profile = DecisionProfile | LLMProfile


class SetupConfig(Section):
    selector: str | None = None
    planner: str | None = None
    executor: str | None = None
    delegate: str | None = None

    @model_validator(mode="after")
    def _roles(self) -> Self:
        if self.selector and self.planner is None and self.executor is None:
            return self
        if self.selector is None and self.planner and self.executor:
            return self
        raise ValueError("Choose either a selector or both a planner and an executor")

    @property
    def action_profile(self) -> str:
        return self.selector or self.executor or ""


class DecisionConfig(Section):
    max_response_repair_attempts: PositiveInt = 3
    system_prompt: str = "assets/prompts/decision/system.jinja2"
    timeout: PositiveFloat = 120
    min_confidence: Annotated[float, Field(ge=0, le=1)] = 0
    max_options: Annotated[int, Field(gt=0, le=252)] = 252
    max_choices: Annotated[int, Field(gt=0, le=255)] = 255
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


class PlanningConfig(Section):
    subgoal_steps: PositiveInt = 10


class AgentConfig(Section):
    repeat_warn_after: PositiveInt = 3
    stuck_fail_after: PositiveInt = 5
    no_progress_note_after: PositiveInt = 3
    no_progress_fail_after: PositiveInt = 5
    cycle_note_after: PositiveInt = 3
    cycle_fail_after: PositiveInt = 4
    max_steps: PositiveInt = DEFAULT_MAX_STEPS
    planning_max_steps: PositiveInt = 40
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    workflow_timeout: PositiveFloat | None = None
    settle_delay: NonNegativeFloat = 0.1
    max_output_validation_attempts: PositiveInt = 2

    system_prompt: str = DEFAULT_FAST_SYSTEM_PROMPT
    parallel_tools: bool = True
    streaming: bool = True
    decision: DecisionConfig = Field(default_factory=DecisionConfig)


class ToolsConfig(Section):
    exclude: list[str] | None = None
    max_schema_combinations: PositiveInt = 18
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


_MEMORY_BLOCKS = {
    "executor": ("setup", "goal", "last_step", "history", "app_cards"),
    "planner": ("setup", "goal", "findings", "last_subgoal", "subgoal_history"),
}


class AgentMemoryConfig(Section):
    blocks: list[str] = Field(default_factory=list)
    history: PositiveInt = 5


class ExecutorMemoryConfig(AgentMemoryConfig):
    last_step_screenshot: bool = True


class MemoryConfig(Section):
    token_limit: PositiveInt = DEFAULT_TOKEN_LIMIT
    insert_method: InsertMethod = InsertMethod.USER
    executor: ExecutorMemoryConfig = Field(
        default_factory=lambda: ExecutorMemoryConfig(blocks=list(_MEMORY_BLOCKS["executor"]))
    )
    planner: AgentMemoryConfig = Field(
        default_factory=lambda: AgentMemoryConfig(blocks=list(_MEMORY_BLOCKS["planner"]))
    )

    @field_validator("executor", "planner", mode="before")
    @classmethod
    def _default_blocks(cls, value: object, info: ValidationInfo) -> object:
        match value:
            case dict() if "blocks" not in value:
                return {"blocks": list(_MEMORY_BLOCKS[info.field_name or ""]), **value}
            case _:
                return value

    @field_validator("executor", "planner")
    @classmethod
    def _known_blocks(cls, value: AgentMemoryConfig, info: ValidationInfo) -> AgentMemoryConfig:
        valid = _MEMORY_BLOCKS[info.field_name or ""]
        for block in value.blocks:
            if block not in valid:
                raise ValueError(f"Unknown memory block {block!r}. Valid names: {', '.join(valid)}")
        return value


class MobileConfig(Section):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    setups: dict[str, SetupConfig] = Field(default_factory=dict)
    default_setup: str | None = None
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    analytics: TelemetryConfig = Field(default_factory=TelemetryConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)

    @model_validator(mode="after")
    def _references(self) -> Self:
        if self.default_setup is not None and self.default_setup not in self.setups:
            raise ValueError(f"Unknown setup {self.default_setup!r} for default_setup")
        for name, setup in self.setups.items():
            for role in ("selector", "executor", "planner", "delegate"):
                reference = getattr(setup, role)
                if reference is None:
                    continue
                if reference not in self.profiles:
                    raise ValueError(f"Unknown profile {reference!r} for {role} in setup {name!r}")
                if role in {"planner", "delegate"} and self.profiles[reference].kind != "llm":
                    raise ValueError(f"Setup {name!r}: {role} must be an LLM profile")
            if setup.delegate and self.profiles[setup.action_profile].kind != "decision":
                raise ValueError(f"Setup {name!r}: a delegate requires a decision model profile")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        # The schema uses the words on/off as strings. PyYAML's default YAML
        # 1.1 resolver would coerce them to booleans before validation.
        with Path(path).open(encoding="utf-8") as stream:
            try:
                data = yaml.load(stream, Loader=_ConfigYAML)  # type: ignore[no-untyped-call]
            except yaml.YAMLError:
                raise ValueError("Invalid agents.yaml syntax") from None
        if type(data) is dict:
            memory = data.get("memory")
            if type(memory) is dict and "blocks" in memory:
                raise OldConfigKeys(
                    "Delete memory.blocks; list blocks under memory.executor.blocks"
                )
            agent = data.get("agent")
            planning = agent.get("planning") if type(agent) is dict else None
            if type(planning) is dict and "max_planner_turns" in planning:
                raise OldConfigKeys(
                    "Delete agent.planning.max_planner_turns from agents.yaml; "
                    "planner turns now count against agent.planning_max_steps."
                )
            if {"llm_profiles", "decision_profiles", "default_" + "llm"} & data.keys() or (
                type(agent) is dict and {"selector", "delegation"} & agent.keys()
            ):
                raise OldConfigKeys(
                    "Config uses old keys; run: mobilerun agent profiles add … / agent configure …"
                )
        try:
            return cls.model_validate({} if data is None else data)
        except ValidationError as exc:
            if message := profile_validation_message(exc):
                raise ValueError(message) from None
            raise


def profile_validation_message(error: ValidationError) -> str | None:
    """Render profile policy errors without any rejected values."""
    for item in error.errors(include_input=False, include_context=False, include_url=False):
        if item["type"] == "profile_credentials":
            return item["msg"]
        if item["type"] == "profile_provider":
            path = item["loc"]
            profile = f" in profile {path[1]!r}" if len(path) > 2 and path[0] == "profiles" else ""
            return f"Unknown provider{profile}; see `mobilerun agent profiles add --help`."
    return None


class OldConfigKeys(ValueError):
    """A removed CLI configuration schema was found."""


class _ConfigYAML(yaml.SafeLoader):
    yaml_implicit_resolvers = {
        initial: [
            (tag, pattern)
            for tag, pattern in rules
            if tag != "tag:yaml.org,2002:bool" or initial in "tTfF"
        ]
        for initial, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

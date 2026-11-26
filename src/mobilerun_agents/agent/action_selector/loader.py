"""Resolve the configured selector and its optional delegate by profile name."""

import os
from dataclasses import dataclass
from typing import Literal

from google.genai.types import HttpOptions
from llama_index.core.llms import LLM
from llama_index.llms.openai.utils import O1_MODELS

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.decision_model.base import DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.delegating import DelegatingSelector
from mobilerun_agents.agent.action_selector.llm.factory import build_llm, capabilities
from mobilerun_agents.agent.action_selector.llm.llama_index import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.auth import CredentialMissing, resolve
from mobilerun_agents.agent.planner import LlamaIndexPlanner, Planner
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.settings import DEFAULT_MODELS
from mobilerun_agents.config.user_config import DecisionProfile, LLMProfile, MobileConfig, Vision

Role = Literal["agent", "executor"]
LLMRole = Literal["agent", "planner", "executor"]


class NoModelConfigured(ValueError):
    """Neither a configured selector nor a provider credential was available."""


@dataclass(frozen=True)
class SetupSpec:
    selector: ActionSelector | None = None
    planner: Planner | None = None
    executor: ActionSelector | None = None


async def _profile_llm(
    config: MobileConfig,
    profile: LLMProfile,
    role: LLMRole,
    api_key: str | None = None,
) -> LLM:
    decision = config.agent.decision
    options: dict[str, object] = {"timeout": decision.timeout}
    if profile.provider == "openai" and profile.model in O1_MODELS:
        effort = (
            profile.reasoning or {"agent": "medium", "planner": "high", "executor": "low"}[role]
        )
        options["reasoning_options"] = {"effort": effort}
    if profile.temperature is not None:
        options["temperature"] = profile.temperature
    if profile.base_url:
        if profile.provider == "gemini":
            options["http_options"] = HttpOptions(base_url=profile.base_url)
        elif profile.provider in {"anthropic", "ollama"}:
            options["base_url"] = profile.base_url
        else:
            options["api_base"] = profile.base_url
    return await build_llm(profile.provider, profile.model, api_key=api_key, **options)


async def _llm_selector(
    config: MobileConfig,
    profile: LLMProfile,
    role: Role,
    api_key: str | None = None,
    vision: Vision | None = None,
) -> ActionSelector:
    llm = await _profile_llm(config, profile, role, api_key)
    caps = capabilities(profile.provider, profile.model)
    selector = LlamaIndexLLMSelector(
        llm,
        vision=vision or profile.vision or ("on" if caps.vision else "off"),
        image_limits=caps,
        native_tool_api=caps.tool_call_format,
        config=config.agent.decision,
    )
    selector.debug_label = f"llm:{profile.provider}/{profile.model}"
    return selector


async def _action_selector(
    config: MobileConfig,
    name: str,
    role: Role,
    delegate: str | None,
    vision: Vision | None,
) -> ActionSelector:
    match config.profiles[name]:
        case DecisionProfile() as profile:
            await resolve(profile.provider)
            decision = config.agent.decision
            model = TypeSafeDecisionModel(
                min_confidence=profile.min_confidence if delegate is None else None, config=decision
            )
            primary = DecisionSelector(model, vision or profile.vision or "on", config=decision)
            primary.debug_label = "jev"
            if delegate is not None:
                fallback = await _llm_selector(
                    config, _llm_profile(config, delegate), role, vision=vision
                )
                if profile.min_confidence is not None:
                    return DelegatingSelector(
                        primary, fallback, min_confidence=profile.min_confidence
                    )
                return DelegatingSelector(primary, fallback)
            return primary
        case LLMProfile() as profile:
            return await _llm_selector(config, profile, role, vision=vision)


def _llm_profile(config: MobileConfig, name: str) -> LLMProfile:
    match config.profiles[name]:
        case LLMProfile() as profile:
            return profile
        case DecisionProfile():
            raise ValueError(f"Profile {name!r} must be an LLM profile")


async def build_setup(name: str | None, vision: Vision | None = None) -> SetupSpec:
    config = ConfigLoader.load()
    selected = name if name is not None else config.default_setup or next(iter(config.setups), None)
    if selected is None:
        for profile in config.profiles.values():
            match profile:
                case LLMProfile():
                    return SetupSpec(
                        selector=await _llm_selector(config, profile, "agent", vision=vision)
                    )
        profile, key = await _credential_profile()
        return SetupSpec(selector=await _llm_selector(config, profile, "agent", key, vision))
    if selected not in config.setups:
        raise ValueError(f"Unknown setup {selected!r}")
    setup = config.setups[selected]
    if setup.planner is None:
        return SetupSpec(
            selector=await _action_selector(
                config, setup.action_profile, "agent", setup.delegate, vision
            )
        )
    llm = await _profile_llm(config, _llm_profile(config, setup.planner), "planner")
    executor = await _action_selector(
        config, setup.action_profile, "executor", setup.delegate, vision
    )
    return SetupSpec(planner=LlamaIndexPlanner(llm), executor=executor)


async def default_setup() -> SetupSpec:
    return await build_setup(None)


async def _credential_profile() -> tuple[LLMProfile, str]:
    for provider, model in DEFAULT_MODELS.items():
        env = MODELS[provider].env_var
        try:
            key = await resolve(provider, os.environ.get(env) if env else None)
        except CredentialMissing:
            continue
        return LLMProfile(provider=provider, model=model), key.api_key
    raise NoModelConfigured("No model configured. Run: mobilerun agent profiles add --help")

"""Resolve the configured selector and its optional delegate by profile name."""

import os
from typing import Literal

from google.genai.types import HttpOptions
from llama_index.core.llms import LLM
from llama_index.llms.openai.utils import O1_MODELS

from mobilerun_agents.agent.action_selector.base import ActionSelector
from mobilerun_agents.agent.action_selector.decision_model.base import DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.loader import load_decision_model
from mobilerun_agents.agent.action_selector.delegating import DelegatingSelector
from mobilerun_agents.agent.action_selector.llm.factory import build_llm, capabilities
from mobilerun_agents.agent.action_selector.llm.llama_index import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.auth import CredentialMissing, resolve
from mobilerun_agents.agent.planner import LlamaIndexPlanner, Planner
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.settings import DEFAULT_MODELS
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig, Vision

Role = Literal["agent", "executor"]
LLMRole = Literal["agent", "planner", "executor"]


class NoModelConfigured(ValueError):
    """Neither a configured selector nor a provider credential was available."""


async def _load_selector(
    config: MobileConfig, name: str, role: Role, vision: Vision | None = None
) -> ActionSelector:
    decision = config.agent.decision
    if name in config.decision_profiles:
        profile = config.decision_profiles[name]
        model = load_decision_model(profile, decision)
        selector = DecisionSelector(model, vision or profile.vision, config=decision)
        selector.debug_label = "jev"
        return selector

    return await _llm_selector(config, config.llm_profiles[name], role, vision=vision)


async def _profile_llm(
    config: MobileConfig,
    profile: LLMProfile,
    role: LLMRole,
    api_key: str | None = None,
) -> LLM:
    decision = config.agent.decision
    options = dict(profile.kwargs)
    options.setdefault("timeout", decision.timeout)
    if api_key is not None:
        options["api_key"] = api_key
    if profile.provider == "openai" and profile.model in O1_MODELS:
        effort = (
            profile.reasoning_effort
            or {"agent": "medium", "planner": "high", "executor": "low"}[role]
        )
        options["reasoning_options"] = {**options.get("reasoning_options", {}), "effort": effort}
    if profile.temperature is not None:
        options.setdefault("temperature", profile.temperature)
    if profile.base_url:
        if profile.provider == "gemini":
            options["http_options"] = HttpOptions.model_validate(
                options.get("http_options", {})
            ).model_copy(update={"base_url": profile.base_url})
        elif profile.provider in {"anthropic", "ollama"}:
            options["base_url"] = profile.base_url
        else:
            options["api_base"] = profile.base_url
    return await build_llm(profile.provider, profile.model, **options)


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


async def build_selector(
    config: MobileConfig, role: Role = "agent", vision: Vision | None = None
) -> ActionSelector:
    delegation = config.agent.delegation
    if delegation is None:
        name = config.agent.selector
        if name == "default" and name not in config.decision_profiles:
            name = config.default_llm or name
        if name in config.decision_profiles or name in config.llm_profiles:
            return await _load_selector(config, name, role, vision)
        if config.default_llm or name != "default":
            raise ValueError(f"Unknown selector profile: {name}")
        profile, key = await _credential_profile()
        return await _llm_selector(config, profile, role, key, vision)
    name = delegation.primary or config.agent.selector
    primary = await _load_selector(config, name, role, vision)
    delegate = (
        primary
        if name == delegation.delegate
        else await _load_selector(config, delegation.delegate, role, vision)
    )
    return DelegatingSelector(primary, delegate, config=delegation)


async def default_selector(role: Role, vision: Vision | None = None) -> ActionSelector:
    return await build_selector(ConfigLoader.load(), role, vision)


async def _credential_profile() -> tuple[LLMProfile, str]:
    for provider, model in DEFAULT_MODELS.items():
        env = MODELS[provider].env_var
        try:
            key = await resolve(provider, os.environ.get(env) if env else None)
        except CredentialMissing:
            continue
        return LLMProfile(provider=provider, model=model), key.api_key
    raise NoModelConfigured("No model configured. Run: mobilerun agent setup")


async def default_planner() -> Planner:
    config = ConfigLoader.load()
    delegation = config.agent.delegation
    primary = (delegation.primary if delegation else None) or config.agent.selector
    fallback = (
        delegation.delegate if delegation and primary in config.decision_profiles else primary
    )
    name = config.default_llm or fallback
    if name in config.llm_profiles:
        llm = await _profile_llm(config, config.llm_profiles[name], "planner")
    else:
        profile, key = await _credential_profile()
        llm = await _profile_llm(config, profile, "planner", key)
    return LlamaIndexPlanner(llm)

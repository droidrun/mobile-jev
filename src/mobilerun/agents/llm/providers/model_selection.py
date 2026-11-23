from __future__ import annotations

from mobilerun.agents.config.user_config import LLMProfile, MobileConfig
from mobilerun.agents.llm.providers.setup_service import (
    SetupSelection,
    apply_selection_to_roles,
)
from mobilerun.agents.llm.providers.types import ProviderVariantSpec

ALL_CONFIG_ROLES = (
    "manager",
    "executor",
    "fast_agent",
    "app_opener",
    "structured_output",
)


def apply_model_selection(
    config: MobileConfig,
    *,
    family_id: str,
    variant: ProviderVariantSpec,
    selected_auth_mode: str,
    selected_model: str,
    selected_api_key: str | None,
    selected_api_key_source: str | None,
    selected_base_url: str | None,
    credential_path: str | None,
) -> None:
    selection = SetupSelection(
        family_id=family_id,
        variant_id=variant.id,
        auth_mode=selected_auth_mode,
        model=selected_model,
        api_key=selected_api_key,
        api_key_source=selected_api_key_source or "auto",
        base_url=selected_base_url,
        credential_path=credential_path,
    )
    apply_selection_to_roles(config, selection, ALL_CONFIG_ROLES)


def set_profile_max_tokens(profile: LLMProfile, value: int) -> None:
    profile.kwargs = dict(profile.kwargs)
    profile.kwargs["max_tokens"] = value


def set_profile_context_window(profile: LLMProfile, value: int) -> None:
    profile.kwargs = dict(profile.kwargs)
    profile.kwargs["context_window"] = value


def any_ollama_profile(config: MobileConfig) -> bool:
    return any(profile.provider == "Ollama" for profile in config.llm_profiles.values())


def any_minimax_profile(config: MobileConfig) -> bool:
    return any(profile.provider_family == "minimax" for profile in config.llm_profiles.values())


def is_vision_enabled(config: MobileConfig) -> bool:
    """Check if vision is enabled on any agent."""
    return (
        config.agent.manager.vision
        or config.agent.executor.vision
        or config.agent.fast_agent.vision
    )


def set_vision_all(config: MobileConfig, enabled: bool) -> None:
    """Set vision on all agents at once."""
    config.agent.manager.vision = enabled
    config.agent.executor.vision = enabled
    config.agent.fast_agent.vision = enabled

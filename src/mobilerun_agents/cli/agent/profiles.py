"""Manage provider profiles while preserving named selector and delegation roles."""

import click

from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.config.user_config import (
    DecisionProfile,
    DelegationConfig,
    LLMProfile,
    MobileConfig,
)


def primary(config: MobileConfig) -> str:
    return (
        config.agent.delegation.primary if config.agent.delegation else None
    ) or config.agent.selector


def default_role(config: MobileConfig) -> str | None:
    role = primary(config)
    if role in config.decision_profiles and config.agent.delegation:
        role = config.agent.delegation.delegate
    return config.default_llm or (role if role in config.llm_profiles else None)


def providers(config: MobileConfig) -> dict[str, str]:
    result: dict[str, str] = {}
    for role, profile in config.llm_profiles.items():
        if profile.provider in result:
            raise click.ClickException(
                f"Multiple profiles use {profile.provider}; choose one in agents.yaml first."
            )
        if profile.provider not in MODELS or role in config.decision_profiles:
            raise click.ClickException(
                "Invalid LLM profile in agents.yaml; check its provider and role."
            )
        result[profile.provider] = role
    for role, decision in config.decision_profiles.items():
        if decision.provider in result:
            raise click.ClickException(
                "Multiple decision profiles use typesafe; choose one in agents.yaml first."
            )
        result[decision.provider] = role
    return result


def set_default(config: MobileConfig, role: str) -> None:
    current = primary(config)
    delegation = config.agent.delegation
    if role in config.decision_profiles:
        llm = default_role(config)
        if config.decision_profiles[role].jev_only:
            config.agent.delegation = None
        else:
            if llm not in config.llm_profiles or llm is None:
                raise click.UsageError("Add an LLM profile first, or choose Jev only (--jev-only).")
            routing = delegation or DelegationConfig()
            routing.primary = None
            routing.delegate = llm
            config.agent.delegation = routing
        config.default_llm = llm
        config.agent.selector = role
        return
    if current in config.decision_profiles:
        if config.decision_profiles[current].jev_only:
            config.default_llm = role
            return
        delegation = delegation or DelegationConfig()
        if delegation.delegate in config.decision_profiles:
            raise click.UsageError(
                "The delegate is a decision profile; choose an LLM delegate first."
            )
        if (
            delegation.delegate not in config.llm_profiles
            or delegation.delegate == config.default_llm
        ):
            delegation.delegate = role
        config.agent.delegation = delegation
    elif delegation and delegation.primary:
        delegation.primary = role
    else:
        config.agent.selector = role
    config.default_llm = role


def configure_jev(config: MobileConfig, only: bool, default: bool) -> DecisionProfile:
    role = providers(config).get("typesafe", "typesafe")
    if role in config.llm_profiles:
        raise click.UsageError("The provider name is already in use by another role.")
    if not only and default_role(config) not in config.llm_profiles:
        raise click.UsageError("Add an LLM profile first, or choose Jev only (--jev-only).")
    profile = config.decision_profiles.setdefault(role, DecisionProfile())
    profile.jev_only = only
    if default or primary(config) == role or default_role(config) is None:
        set_default(config, role)
    return profile


def configure(config: MobileConfig, provider: str, model: str, default: bool) -> LLMProfile:
    if not model or any(c.isspace() for c in model):
        raise click.UsageError("Expected a model identifier without spaces.")
    activate = default or default_role(config) is None
    role = providers(config).get(provider, provider)
    profile = config.llm_profiles.get(role)
    if role in config.decision_profiles or (profile and profile.provider != provider):
        raise click.UsageError("The provider name is already in use by another role.")
    if profile is None:
        profile = LLMProfile(provider=provider, model=model)
        config.llm_profiles[role] = profile
    else:
        profile.model = model
    if activate:
        set_default(config, role)
    return profile


def remove(config: MobileConfig, provider: str) -> None:
    role = providers(config)[provider]
    was_default = default_role(config) == role
    delegation = config.agent.delegation
    if role in config.decision_profiles:
        if primary(config) == role:
            config.agent.selector = default_role(config) or "default"
            config.agent.delegation = None
        elif delegation and delegation.delegate == role:
            raise click.UsageError("Change the pinned delegate before removing it.")
        del config.decision_profiles[role]
        return
    if (
        delegation
        and delegation.delegate == role
        and (not was_default or primary(config) not in config.decision_profiles)
    ):
        raise click.ClickException(
            "This provider is a pinned delegate. Change the delegate before removing it."
        )
    del config.llm_profiles[role]
    if was_default:
        if config.llm_profiles:
            set_default(config, next(iter(config.llm_profiles)))
        else:
            if primary(config) in config.decision_profiles:
                config.agent.selector = primary(config)
            else:
                config.agent.selector = "default"
            config.agent.delegation = None
            config.default_llm = None

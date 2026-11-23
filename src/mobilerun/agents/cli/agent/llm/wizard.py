from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import click
from mobilerun.cli.base import render
from mobilerun.cli.base.prompts import (
    SelectChoice,
    select_prompt,
    text_prompt,
)
from rich.console import Console

from mobilerun.agents.clients.oauth.credentials import oauth_credentials_present
from mobilerun.agents.clients.oauth.profiles import (
    load_env_key_sources,
    resolve_env_key,
)
from mobilerun.agents.config.loader import ConfigLoader
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.llm.providers.minimax import (
    MINIMAX_CHINA_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
)
from mobilerun.agents.llm.providers.model_selection import (
    ALL_CONFIG_ROLES,
    any_minimax_profile,
    any_ollama_profile,
    apply_model_selection,
    is_vision_enabled,
    set_profile_context_window,
    set_profile_max_tokens,
    set_vision_all,
)
from mobilerun.agents.llm.providers.registry import (
    VARIANT_ENV_KEY_SLOT,
    model_display_name_for_variant,
    resolve_provider_variant,
)
from mobilerun.agents.llm.providers.setup_service import (
    auth_mode_choices,
    family_choices,
    variant_models,
)
from mobilerun.agents.llm.providers.types import ProviderFamilySpec

_BACK = "__back__"
_CUSTOM_MINIMAX_BASE_URL = "__custom_minimax_base_url__"


@dataclass
class ConfigureWizardCallbacks:
    run_openai_oauth_login: Callable[..., None]
    run_anthropic_oauth_login: Callable[..., None]
    run_gemini_oauth_login: Callable[..., None]
    run_grok_oauth_login: Callable[..., None] | None = None


@dataclass
class ConfigureWizardState:
    family_id: str | None = None
    selected_auth_mode: str | None = None
    selected_model: str | None = None
    selected_api_key: str | None = None
    selected_api_key_source: str | None = None
    selected_base_url: str | None = None
    last_variant_id: str | None = None
    prepared_auth_variant_id: str | None = None
    used_advanced_settings: bool = False


def _select_with_back(
    message: str,
    choices: list[SelectChoice],
    *,
    default: str | None = None,
    include_back: bool = True,
) -> str:
    selected = select_prompt(
        message,
        ([*choices, SelectChoice(value=_BACK, label="Back")] if include_back else choices),
        default=default,
    )
    if selected is None:
        raise click.Abort()
    return selected


def _print_configure_intro(console: Console) -> None:
    render.kv(
        [
            ("Mobilerun Configure", "Choose your provider, auth method, and model"),
            ("Advanced settings", "Optional"),
        ]
    )


def _print_configure_summary(
    console: Console,
    *,
    family_id: str,
    auth_mode: str,
    provider_label: str,
    variant_id: str,
    model: str,
    used_advanced_settings: bool,
) -> None:
    advanced_line = "Yes" if used_advanced_settings else "No"
    provider_detail = "" if provider_label == "XAI" else f" ({variant_id})"
    model_label = model_display_name_for_variant(family_id, auth_mode, model)
    render.success("Configuration saved")
    render.kv(
        [
            ("Provider", f"{provider_label}{provider_detail}"),
            ("Model", model_label),
            ("Advanced settings changed", advanced_line),
        ]
    )


def _prompt_int(console: Console, message: str, default: int) -> int:
    while True:
        value = text_prompt(message, default=str(default))
        try:
            return int(value)
        except ValueError:
            render.error("Please enter a whole number.")


def _prompt_float(console: Console, message: str, default: float) -> float:
    while True:
        value = text_prompt(message, default=str(default))
        try:
            return float(value)
        except ValueError:
            render.error("Please enter a number.")


def _prompt_model_choice(
    models: list[str],
    *,
    family_id: str,
    auth_mode: str,
    default_model: str,
    allow_back: bool = True,
) -> str:
    if models:
        choice = _select_with_back(
            "Choose model",
            [
                *[
                    SelectChoice(
                        value=item,
                        label=model_display_name_for_variant(family_id, auth_mode, item),
                    )
                    for item in models
                ],
                SelectChoice(
                    value="enter_model",
                    label="Enter custom model",
                ),
            ],
            default=default_model or None,
            include_back=allow_back,
        )
        if choice in {_BACK, "enter_model"}:
            if choice == _BACK:
                return _BACK
            return text_prompt("Model", default=default_model)
        return choice

    choice = _select_with_back(
        "Choose model",
        [
            SelectChoice(
                value="enter_model",
                label="Enter custom model",
            )
        ],
        default="enter_model",
        include_back=allow_back,
    )
    if choice == _BACK:
        return _BACK
    return text_prompt("Model", default=default_model)


def _prompt_api_key_source(variant: Any) -> str:
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    sources = load_env_key_sources().get(env_slot) if env_slot else None
    choices: list[SelectChoice] = []
    if sources and sources.shell:
        choices.append(SelectChoice(value="env", label="Use env key"))
    if sources and sources.saved:
        choices.append(SelectChoice(value="file", label="Use saved key"))
    choices.append(SelectChoice(value="paste", label="Paste new key"))
    return _select_with_back("API key source", choices, default=choices[0].value)


def _prompt_api_key_for_variant(variant: Any) -> tuple[str, str]:
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    if not env_slot:
        return text_prompt("API key", secret=True), "file"

    source = _prompt_api_key_source(variant)
    if source == _BACK:
        return _BACK, _BACK
    if source == "env":
        return resolve_env_key(env_slot, "env"), "env"
    if source == "file":
        return resolve_env_key(env_slot, "file"), "file"
    return text_prompt("API key", secret=True), "file"


def _prompt_base_url_for_variant(variant: Any) -> str:
    if variant.id != "MiniMax":
        return text_prompt("Base URL", default=variant.base_url or "", secret=False)

    selected = select_prompt(
        "Choose MiniMax API region",
        [
            SelectChoice(
                value=MINIMAX_GLOBAL_BASE_URL,
                label="Global",
                hint=MINIMAX_GLOBAL_BASE_URL,
            ),
            SelectChoice(
                value=MINIMAX_CHINA_BASE_URL,
                label="Mainland China",
                hint=MINIMAX_CHINA_BASE_URL,
            ),
            SelectChoice(
                value=_CUSTOM_MINIMAX_BASE_URL,
                label="Custom endpoint",
            ),
        ],
        default=MINIMAX_GLOBAL_BASE_URL,
    )
    if selected is None:
        raise click.Abort()
    if selected == _CUSTOM_MINIMAX_BASE_URL:
        return text_prompt("Custom MiniMax base URL", secret=False)
    return selected


def _prompt_oauth_credential_action(credential_path: str) -> str:
    return _select_with_back(
        "OAuth credentials found",
        [
            SelectChoice(value="use_existing", label="Use existing login"),
            SelectChoice(value="login_again", label="Log in again"),
        ],
        default="use_existing",
    )


def _prepare_variant_auth(
    *,
    callbacks: ConfigureWizardCallbacks,
    variant: Any,
    credential_path: str | None,
    selected_model: str,
) -> None:
    if variant.id == "openai_oauth" and credential_path:
        callbacks.run_openai_oauth_login(credential_path=credential_path, model=selected_model)
    elif variant.id == "anthropic_oauth" and credential_path:
        callbacks.run_anthropic_oauth_login(credential_path=credential_path)
    elif variant.id == "gemini_oauth_code_assist" and credential_path:
        callbacks.run_gemini_oauth_login(credential_path=credential_path, model=selected_model)
    elif variant.id == "xai_oauth" and credential_path:
        if callbacks.run_grok_oauth_login is None:
            raise RuntimeError("XAI OAuth login callback is not configured.")
        callbacks.run_grok_oauth_login(
            credential_path=credential_path,
            model=selected_model,
        )


def _configure_advanced_settings(
    console: Console,
    config: MobileConfig,
) -> None:
    default_selection = "vision"
    while True:
        vision_on = is_vision_enabled(config)
        reasoning_on = config.agent.reasoning

        choices = [
            SelectChoice(
                value="vision",
                label=f"Vision {'[ON]' if vision_on else '[OFF]'}",
            ),
            SelectChoice(
                value="reasoning",
                label=f"Reasoning {'[ON]' if reasoning_on else '[OFF]'}",
            ),
            SelectChoice(
                value="max_steps",
                label="Maximum steps",
            ),
            SelectChoice(
                value="temperature",
                label="Temperature",
            ),
            SelectChoice(
                value="max_tokens",
                label="Max tokens",
            ),
        ]
        if any_ollama_profile(config):
            choices.append(
                SelectChoice(
                    value="context_window",
                    label="Context window (Ollama)",
                )
            )
        choices.append(SelectChoice(value="done", label="Done"))

        selected = _select_with_back(
            "Advanced settings",
            choices,
            default=default_selection,
        )

        if selected in {_BACK, "done"}:
            return

        if selected == "vision":
            set_vision_all(config, not vision_on)
        elif selected == "reasoning":
            config.agent.reasoning = not reasoning_on
        elif selected == "max_steps":
            config.agent.max_steps = _prompt_int(
                console, "Maximum steps", default=config.agent.max_steps
            )
        elif selected == "temperature":
            default_temp = config.llm_profiles[ALL_CONFIG_ROLES[0]].temperature
            while True:
                value = _prompt_float(console, "Temperature", default=default_temp)
                if not any_minimax_profile(config) or 0 < value <= 1:
                    break
                render.error("MiniMax temperature must be greater than 0 and no more than 1.")
            for role in ALL_CONFIG_ROLES:
                if role in config.llm_profiles:
                    config.llm_profiles[role].temperature = value
        elif selected == "max_tokens":
            current_value = config.llm_profiles[ALL_CONFIG_ROLES[0]].kwargs.get("max_tokens", 1024)
            try:
                current_default = int(current_value)
            except (TypeError, ValueError):
                current_default = 1024
            value = _prompt_int(console, "Max tokens", default=current_default)
            for role in ALL_CONFIG_ROLES:
                if role in config.llm_profiles:
                    set_profile_max_tokens(config.llm_profiles[role], value)
        elif selected == "context_window":
            current_value = config.llm_profiles[ALL_CONFIG_ROLES[0]].kwargs.get(
                "context_window", 32768
            )
            try:
                current_default = int(current_value)
            except (TypeError, ValueError):
                current_default = 32768
            value = _prompt_int(console, "Context window (-1 = model max)", default=current_default)
            for role in ALL_CONFIG_ROLES:
                if role in config.llm_profiles:
                    set_profile_context_window(config.llm_profiles[role], value)

        default_selection = selected


def _configure_provider_model(
    console: Console,
    config: MobileConfig,
    callbacks: ConfigureWizardCallbacks,
    state: ConfigureWizardState,
    families: tuple[ProviderFamilySpec, ...],
    family_labels: dict[str, str],
    *,
    provider_is_fixed: bool,
    auth_mode_is_fixed: bool,
    model_is_fixed: bool,
    api_key: str | None,
    base_url: str | None,
) -> bool:
    """Run the provider → auth → model → credentials flow.

    Returns True when the flow completes and the selection has been applied
    to *config*.  Returns False when the user backs out entirely.
    """
    while True:
        # --- Provider family ---
        if state.family_id is None:
            state.family_id = _select_with_back(
                "Choose provider",
                [SelectChoice(value=family.id, label=family.display_name) for family in families],
                default="gemini",
                include_back=True,
            )
            if state.family_id == _BACK:
                state.family_id = None
                return False
        render.kv([("Selected provider family", family_labels[state.family_id])])

        # --- Auth mode ---
        modes = auth_mode_choices(state.family_id)
        if auth_mode_is_fixed:
            state.selected_auth_mode = click.Choice(list(modes), case_sensitive=False).convert(
                state.selected_auth_mode or "", None, None
            )
        else:
            while True:
                if len(modes) == 1:
                    state.selected_auth_mode = modes[0]
                    break
                state.selected_auth_mode = _select_with_back(
                    "Choose auth mode",
                    [SelectChoice(value=mode, label=mode.replace("_", " ")) for mode in modes],
                    default=modes[0],
                )
                if state.selected_auth_mode == _BACK:
                    state.family_id = None
                    break
                break
            if state.selected_auth_mode is None:
                continue
            if state.selected_auth_mode == _BACK:
                state.family_id = None
                state.selected_auth_mode = None
                continue

        # --- Model ---
        if state.family_id is None or state.selected_auth_mode is None:
            raise click.ClickException("Provider and auth mode are required")
        models = list(variant_models(state.family_id, state.selected_auth_mode))
        variant = resolve_provider_variant(state.family_id, state.selected_auth_mode)
        default_model = models[0] if models else (variant.default_model or "")

        if not model_is_fixed:
            while True:
                state.selected_model = _prompt_model_choice(
                    models,
                    family_id=state.family_id,
                    auth_mode=state.selected_auth_mode,
                    default_model=default_model,
                )
                if state.selected_model == _BACK:
                    state.selected_model = None
                    if provider_is_fixed and (auth_mode_is_fixed or len(modes) == 1):
                        return False
                    if auth_mode_is_fixed or len(modes) == 1:
                        if not provider_is_fixed:
                            state.family_id = None
                        if not auth_mode_is_fixed:
                            state.selected_auth_mode = None
                    else:
                        state.selected_auth_mode = None
                    break
                break
            if state.selected_model is None:
                continue

        # --- Credentials ---
        credential_path: str | None = variant.credential_path

        if variant.id != state.last_variant_id:
            if api_key is None:
                state.selected_api_key = None
                state.selected_api_key_source = None
            if base_url is None:
                state.selected_base_url = None
            state.last_variant_id = variant.id
            state.prepared_auth_variant_id = None

        non_interactive = provider_is_fixed and model_is_fixed

        if variant.requires_api_key and not state.selected_api_key:
            if non_interactive:
                # Auto-resolve key from saved credentials
                env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
                if env_slot:
                    key = resolve_env_key(env_slot, "auto")
                    if key:
                        state.selected_api_key = key
                        state.selected_api_key_source = "auto"
                    else:
                        raise click.ClickException(
                            f"No API key found for {variant.id}. Pass --api-key or save one first."
                        )
                else:
                    raise click.ClickException(
                        f"No API key provided for {variant.id}. Pass --api-key."
                    )
            else:
                selected_key, selected_source = _prompt_api_key_for_variant(variant)
                if selected_key == _BACK:
                    if model_is_fixed:
                        return False
                    state.selected_model = None
                    continue
                state.selected_api_key = selected_key
                state.selected_api_key_source = selected_source
        if variant.requires_base_url and not state.selected_base_url:
            if non_interactive and variant.base_url:
                state.selected_base_url = variant.base_url
            else:
                state.selected_base_url = _prompt_base_url_for_variant(variant)
        if (
            credential_path
            and variant.auth_mode == "oauth"
            and state.prepared_auth_variant_id != variant.id
            and oauth_credentials_present(credential_path, variant.id)
        ):
            if non_interactive:
                state.prepared_auth_variant_id = variant.id
            else:
                oauth_action = _prompt_oauth_credential_action(credential_path)
                if oauth_action == _BACK:
                    if model_is_fixed:
                        return False
                    state.selected_model = None
                    continue
                if oauth_action == "use_existing":
                    state.prepared_auth_variant_id = variant.id

        # --- Apply ---
        if (
            state.family_id is None
            or state.selected_auth_mode is None
            or state.selected_model is None
        ):
            raise click.ClickException("Provider, auth mode and model are required")
        if credential_path and state.prepared_auth_variant_id != variant.id:
            _prepare_variant_auth(
                callbacks=callbacks,
                variant=variant,
                credential_path=credential_path,
                selected_model=state.selected_model,
            )
            state.prepared_auth_variant_id = variant.id

        apply_model_selection(
            config,
            family_id=state.family_id,
            variant=variant,
            selected_auth_mode=state.selected_auth_mode,
            selected_model=state.selected_model,
            selected_api_key=state.selected_api_key,
            selected_api_key_source=state.selected_api_key_source,
            selected_base_url=state.selected_base_url,
            credential_path=credential_path,
        )
        return True


def run_configure_wizard(
    console: Console,
    callbacks: ConfigureWizardCallbacks,
    *,
    provider: str | None,
    auth_mode: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    config = ConfigLoader.load()
    _print_configure_intro(console)

    families = family_choices()
    family_ids = [family.id for family in families]
    family_labels = {family.id: family.display_name for family in families}
    state = ConfigureWizardState(
        selected_api_key=api_key,
        selected_api_key_source="file" if api_key else None,
        selected_base_url=base_url,
    )

    provider_is_fixed = provider is not None
    auth_mode_is_fixed = auth_mode is not None
    model_is_fixed = model is not None

    if provider_is_fixed:
        state.family_id = click.Choice(family_ids, case_sensitive=False).convert(
            provider, None, None
        )
    if auth_mode_is_fixed:
        state.selected_auth_mode = auth_mode
    if model_is_fixed:
        state.selected_model = model

    # Enter a partially fixed provider flow only when another flag establishes
    # where that flow should resume. A provider-only invocation starts at the
    # top-level menu, matching the fully interactive wizard.
    provider_configured = False
    if provider_is_fixed and (auth_mode_is_fixed or model_is_fixed):
        provider_configured = _configure_provider_model(
            console,
            config,
            callbacks,
            state,
            families,
            family_labels,
            provider_is_fixed=provider_is_fixed,
            auth_mode_is_fixed=auth_mode_is_fixed,
            model_is_fixed=model_is_fixed,
            api_key=api_key,
            base_url=base_url,
        )
        if provider_configured:
            ConfigLoader.save(config)
            if state.family_id is None:
                raise click.ClickException("A provider is required")
            _print_configure_summary(
                console,
                family_id=state.family_id or "",
                auth_mode=state.selected_auth_mode or "",
                provider_label=family_labels[state.family_id],
                variant_id=state.last_variant_id or state.family_id,
                model=state.selected_model or "",
                used_advanced_settings=False,
            )
            return

    # --- Top-level menu ---
    while True:
        action = select_prompt(
            "Configure",
            [
                SelectChoice(value="provider_model", label="Provider / Model"),
                SelectChoice(value="advanced", label="Advanced settings"),
                SelectChoice(value="finish", label="Finish"),
            ],
            default="finish" if provider_configured else "provider_model",
        )

        if action is None:
            return
        if action == "provider_model":
            # Reset provider state so the user can re-pick
            state.family_id = None
            state.selected_auth_mode = None
            state.selected_model = None
            state.selected_api_key = api_key
            state.selected_api_key_source = "file" if api_key else None
            state.selected_base_url = base_url
            state.last_variant_id = None
            state.prepared_auth_variant_id = None

            completed = _configure_provider_model(
                console,
                config,
                callbacks,
                state,
                families,
                family_labels,
                provider_is_fixed=False,
                auth_mode_is_fixed=False,
                model_is_fixed=False,
                api_key=api_key,
                base_url=base_url,
            )
            if completed:
                provider_configured = True

        elif action == "advanced":
            state.used_advanced_settings = True
            _configure_advanced_settings(console, config)

        elif action == "finish":
            ConfigLoader.save(config)
            if provider_configured and state.family_id:
                _print_configure_summary(
                    console,
                    family_id=state.family_id,
                    auth_mode=state.selected_auth_mode or "",
                    provider_label=family_labels[state.family_id],
                    variant_id=state.last_variant_id or state.family_id,
                    model=state.selected_model or "",
                    used_advanced_settings=state.used_advanced_settings,
                )
            else:
                render.success("Configuration saved.")
            return

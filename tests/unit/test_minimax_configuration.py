import pytest

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.llm.llm_picker import load_llm
from mobilerun.agents.llm.providers.minimax import (
    MINIMAX_CHINA_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
    MINIMAX_LEGACY_BASE_URL,
)
from mobilerun.agents.llm.providers.registry import resolve_provider_variant
from mobilerun.agents.llm.providers.setup_service import (
    SetupSelection,
    apply_selection_to_roles,
    create_profile_for_variant,
)


def test_minimax_registry_uses_current_global_endpoint() -> None:
    variant = resolve_provider_variant("minimax", "api_key")

    assert variant.base_url == MINIMAX_GLOBAL_BASE_URL
    assert variant.default_model == "MiniMax-M3"
    assert variant.models == (
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    )
    assert MINIMAX_CHINA_BASE_URL == "https://api.minimaxi.com/v1"
    assert MINIMAX_LEGACY_BASE_URL == "https://api.minimaxi.chat/v1"


def test_minimax_setup_profile_uses_openai_like_transport() -> None:
    variant = resolve_provider_variant("minimax", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="minimax",
            variant_id=variant.id,
            auth_mode="api_key",
            model="MiniMax-M2.7",
            api_key_source="env",
        ),
    )

    assert profile.provider == "OpenAILike"
    assert profile.provider_family == "minimax"
    assert profile.base_url == MINIMAX_GLOBAL_BASE_URL
    assert profile.api_base == MINIMAX_GLOBAL_BASE_URL
    assert profile.kwargs == {}


def test_minimax_issue_style_generation_normalizes_every_role_temperature() -> None:
    config = MobileConfig()
    roles = tuple(config.llm_profiles)
    selection = SetupSelection(
        family_id="minimax",
        variant_id="MiniMax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        api_key="env-test-key",
        api_key_source="env",
    )

    apply_selection_to_roles(config, selection, roles)

    assert set(config.llm_profiles) == {
        "manager",
        "executor",
        "fast_agent",
        "app_opener",
        "structured_output",
    }
    for role, profile in config.llm_profiles.items():
        assert profile.provider == "OpenAILike", role
        assert profile.provider_family == "minimax", role
        assert profile.model == "MiniMax-M2.7", role
        assert profile.base_url == MINIMAX_GLOBAL_BASE_URL, role
        assert profile.api_base == MINIMAX_GLOBAL_BASE_URL, role
        assert 0 < profile.temperature <= 1, role

    assert config.llm_profiles["manager"].temperature == 0.2
    assert config.llm_profiles["executor"].temperature == 0.1
    assert config.llm_profiles["fast_agent"].temperature == 0.2
    assert config.llm_profiles["app_opener"].temperature == 0.1
    assert config.llm_profiles["structured_output"].temperature == 0.1


def test_minimax_fast_agent_fallback_normalizes_hidden_role_temperatures() -> None:
    config = MobileConfig()
    selection = SetupSelection(
        family_id="minimax",
        variant_id="MiniMax",
        auth_mode="api_key",
        model="MiniMax-M2.7",
        api_key="env-test-key",
        api_key_source="env",
    )

    apply_selection_to_roles(config, selection, ("fast_agent",))

    assert config.llm_profiles["app_opener"].temperature == 0.1
    assert config.llm_profiles["structured_output"].temperature == 0.1


def test_non_minimax_generated_profile_preserves_zero_temperature() -> None:
    variant = resolve_provider_variant("gemini", "api_key")
    profile = create_profile_for_variant(
        variant,
        SetupSelection(
            family_id="gemini",
            variant_id=variant.id,
            auth_mode="api_key",
            model=variant.default_model or "",
            api_key_source="env",
        ),
        temperature=0.0,
    )

    assert profile.temperature == 0.0


def test_legacy_minimax_alias_defaults_to_global_endpoint() -> None:
    llm = load_llm("MiniMax", model="MiniMax-M2.7", api_key="stub")

    assert type(llm).__name__ == "OpenAILike"
    assert llm.api_base == MINIMAX_GLOBAL_BASE_URL


def test_legacy_minimax_alias_uses_minimax_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    llm = load_llm("MiniMax", model="MiniMax-M2.7")

    assert llm.api_key == "minimax-env-key"


def test_legacy_minimax_alias_prefers_explicit_api_key(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")

    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="explicit-minimax-key",
    )

    assert llm.api_key == "explicit-minimax-key"


def test_legacy_minimax_alias_empty_explicit_key_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-env-key")

    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="",
    )

    assert llm.api_key == "minimax-env-key"


def test_legacy_minimax_alias_never_falls_back_to_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")

    with pytest.raises(
        ValueError,
        match="Pass api_key explicitly or set MINIMAX_API_KEY",
    ):
        load_llm("MiniMax", model="MiniMax-M2.7")


def test_legacy_minimax_alias_honors_base_url_override() -> None:
    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="stub",
        base_url=MINIMAX_CHINA_BASE_URL,
    )

    assert llm.api_base == MINIMAX_CHINA_BASE_URL


def test_legacy_minimax_alias_prefers_explicit_api_base() -> None:
    custom_api_base = "https://gateway.example/v1"
    llm = load_llm(
        "MiniMax",
        model="MiniMax-M2.7",
        api_key="stub",
        base_url=MINIMAX_CHINA_BASE_URL,
        api_base=custom_api_base,
    )

    assert llm.api_base == custom_api_base

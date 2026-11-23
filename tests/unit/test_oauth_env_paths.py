import pytest

from mobilerun.agents.clients.oauth.anthropic import AnthropicOAuthSession
from mobilerun.agents.clients.oauth.gemini import GeminiOAuthSession
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthCredentials,
    GrokOAuthCredentialStore,
)
from mobilerun.agents.clients.oauth.openai import (
    OpenAIOAuthCredentials,
    OpenAIOAuthCredentialStore,
)
from mobilerun.agents.config.env import OAuthCredentialPaths
from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.llm.providers.registry import resolve_provider_variant
from mobilerun.agents.llm.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_OAUTH_CREDENTIAL_PATH",
        "ANTHROPIC_OAUTH_CREDENTIAL_PATH",
        "GEMINI_OAUTH_CREDENTIAL_PATH",
        "GROK_OAUTH_CREDENTIAL_PATH",
    ],
)
def test_credential_path_fields_match_environment(monkeypatch, tmp_path, name):
    target = tmp_path / "credentials.json"
    monkeypatch.setenv(name, str(target))
    field = OAuthCredentialPaths.model_fields[name]
    assert field.validation_alias is None
    assert field.description
    assert getattr(OAuthCredentialPaths(), name) == target


@pytest.mark.parametrize(
    ("name", "property_name", "kind"),
    [
        ("OPENAI_OAUTH_CREDENTIAL_PATH", "openai_oauth_credential_path", "openai"),
        (
            "ANTHROPIC_OAUTH_CREDENTIAL_PATH",
            "anthropic_oauth_credential_path",
            "anthropic",
        ),
        ("GEMINI_OAUTH_CREDENTIAL_PATH", "gemini_oauth_credential_path", "gemini"),
        ("GROK_OAUTH_CREDENTIAL_PATH", "grok_oauth_credential_path", "grok"),
    ],
)
def test_credential_paths_load_save_and_precedence(
    monkeypatch, tmp_path, name, property_name, kind
):
    monkeypatch.delenv(name, raising=False)
    assert (
        getattr(get_agent_settings(), property_name)
        == tmp_path / "config/credentials/auth-profiles.json"
    )
    target = tmp_path / "env.json"
    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv(name, str(target))
    assert getattr(get_agent_settings(), property_name) == target

    if kind in ("openai", "grok"):
        store_class, credential_class = {
            "openai": (OpenAIOAuthCredentialStore, OpenAIOAuthCredentials),
            "grok": (GrokOAuthCredentialStore, GrokOAuthCredentials),
        }[kind]
        store_class().save(
            credential_class(access_token="test-token", refresh_token=None, expires_at_ms=None)
        )
        assert store_class().load().access_token == "test-token"
        store_class(explicit).save(
            credential_class(access_token="explicit-token", refresh_token=None, expires_at_ms=None)
        )
        assert store_class(explicit).load().access_token == "explicit-token"
    else:
        session_class = {
            "anthropic": AnthropicOAuthSession,
            "gemini": GeminiOAuthSession,
        }[kind]
        session = session_class(access_token="test-token")
        session._persist_credentials()
        assert session_class()._cached_access_token == "test-token"
        session = session_class(credential_path=str(explicit), access_token="explicit-token")
        session._persist_credentials()
        assert session_class(credential_path=str(explicit))._cached_access_token == "explicit-token"
    assert target.is_file() and explicit.is_file()


@pytest.mark.parametrize(
    ("family", "name"),
    [
        ("openai", "OPENAI_OAUTH_CREDENTIAL_PATH"),
        ("anthropic", "ANTHROPIC_OAUTH_CREDENTIAL_PATH"),
        ("gemini", "GEMINI_OAUTH_CREDENTIAL_PATH"),
        ("xai", "GROK_OAUTH_CREDENTIAL_PATH"),
    ],
)
def test_wizard_registry_path_and_explicit_selection(monkeypatch, tmp_path, family, name):
    monkeypatch.delenv(name, raising=False)
    default = str(tmp_path / "config/credentials/auth-profiles.json")
    assert resolve_provider_variant(family, "oauth").credential_path == default
    target = str(tmp_path / "env.json")
    monkeypatch.setenv(name, target)
    variant = resolve_provider_variant(family, "oauth")
    assert variant.credential_path == target
    selection = SetupSelection(
        family_id=family,
        variant_id=variant.id,
        auth_mode="oauth",
        model=variant.default_model,
        credential_path=str(tmp_path / "explicit.json"),
    )
    assert (
        create_profile_for_variant(variant, selection).credential_path == selection.credential_path
    )

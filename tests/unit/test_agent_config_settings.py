"""Shared device-auth protocol values and credential locations."""

from mobilerun.config import settings

from mobilerun.agents.config import settings as agent_settings


def test_existing_credential_locations_stay_unchanged():
    snapshot = settings.get_settings()
    agent_snapshot = agent_settings.get_agent_settings()
    assert (
        agent_snapshot.oauth_credential_dir
        == agent_snapshot.MOBILERUN_USER_CONFIG_DIR / "credentials"
    )
    assert (
        agent_snapshot.auth_profiles_path
        == agent_snapshot.oauth_credential_dir / "auth-profiles.json"
    )
    assert (
        snapshot.mobilerun_cloud_credential_path
        == agent_snapshot.oauth_credential_dir / "mobilerun-cloud.json"
    )


def test_provider_paths_share_the_existing_auth_profile():
    snapshot = agent_settings.get_agent_settings()
    assert snapshot.openai_oauth_credential_path == snapshot.auth_profiles_path
    assert snapshot.anthropic_oauth_credential_path == snapshot.auth_profiles_path
    assert snapshot.gemini_oauth_credential_path == snapshot.auth_profiles_path
    assert snapshot.grok_oauth_credential_path == snapshot.auth_profiles_path

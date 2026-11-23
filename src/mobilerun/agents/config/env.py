"""Agent-specific OAuth credential file paths."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class OAuthCredentialPaths(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        env_ignore_empty=True,
        use_attribute_docstrings=True,
    )
    OPENAI_OAUTH_CREDENTIAL_PATH: Path | None = None
    """OpenAI OAuth credential file override."""

    ANTHROPIC_OAUTH_CREDENTIAL_PATH: Path | None = None
    """Anthropic OAuth credential file override."""

    GEMINI_OAUTH_CREDENTIAL_PATH: Path | None = None
    """Gemini OAuth credential file override."""

    GROK_OAUTH_CREDENTIAL_PATH: Path | None = None
    """Grok OAuth credential file override."""


class AgentSecrets(BaseSettings):
    """Masked agent credentials, read from exact environment names."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    OPENAI_API_KEY: SecretStr | None = None
    """Openai api key."""

    ANTHROPIC_API_KEY: SecretStr | None = None
    """Anthropic api key."""

    GOOGLE_API_KEY: SecretStr | None = None
    """Google api key."""

    GEMINI_API_KEY: SecretStr | None = None
    """Gemini api key."""

    XAI_API_KEY: SecretStr | None = None
    """Xai api key."""

    DEEPSEEK_API_KEY: SecretStr | None = None
    """Deepseek api key."""

    ZAI_API_KEY: SecretStr | None = None
    """Zai api key."""

    MINIMAX_API_KEY: SecretStr | None = None
    """Minimax api key."""

    ANTHROPIC_OAUTH_TOKEN: SecretStr | None = None
    """Anthropic oauth token."""

    GEMINI_OAUTH_ACCESS_TOKEN: SecretStr | None = None
    """Gemini oauth access token."""

    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    """Langfuse public key."""

    LANGFUSE_SECRET_KEY: SecretStr | None = None
    """Langfuse secret key."""


def get_agent_secrets() -> AgentSecrets:
    """Read a fresh snapshot of agent credentials."""
    return AgentSecrets()

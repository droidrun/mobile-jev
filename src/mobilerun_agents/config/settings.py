"""Agent-owned settings, read fresh at call time using exact environment field names."""

from pathlib import Path

from mobilerun.config.settings import get_settings
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_FAST_SYSTEM_PROMPT = "assets/prompts/fast_agent/system.jinja2"
DEFAULT_LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"
DEFAULT_MAX_STEPS = 25
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
    "openrouter": "openai/gpt-5-mini",
}


class _Environment(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )


class AgentSettings(_Environment):
    MOBILERUN_AGENT_RUNS_DIR: str = ""
    """Parent directory for agent run checkpoints and diagnostics."""

    MOBILERUN_AGENT_IMAGES_DIR: str = ""
    """Parent directory for agent run screenshots."""

    @property
    def runs_dir(self) -> Path:
        return self._directory(self.MOBILERUN_AGENT_RUNS_DIR, "runs")

    @property
    def images_dir(self) -> Path:
        return self._directory(self.MOBILERUN_AGENT_IMAGES_DIR, "images")

    @staticmethod
    def _directory(value: str, default: str) -> Path:
        return (
            Path(value).expanduser()
            if value
            else get_settings().MOBILERUN_USER_CONFIG_DIR / default
        ).absolute()

    MOBILERUN_FINALIZE_TELEMETRY_SECONDS: float = 10.0
    """Finalize telemetry seconds."""

    MOBILERUN_OLLAMA_DEFAULT_CONTEXT_WINDOW: int = 32768
    """Ollama default context window."""

    MOBILERUN_POSTHOG_HOST: str = "https://eu.i.posthog.com"
    """Posthog host."""

    MOBILERUN_POSTHOG_PROJECT_API_KEY: str = "phc_XyD3HKIsetZeRkmnfaBughs8fXWYArSUFc30C0HmRiO"  # fmt: skip
    """Posthog project api key."""

    MOBILERUN_MINIMAX_BASE_URL_GLOBAL: str = "https://api.minimax.io/v1"
    """Minimax base url global."""

    MOBILERUN_DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    """Deepseek base url."""

    MOBILERUN_ZAI_BASE_URL: str = "https://api.z.ai/api/paas/v4"
    """Zai base url."""

    MOBILERUN_OLLAMA_BASE_URL: str = "http://localhost:11434"
    """Ollama base url."""

    MOBILERUN_GROK_BASE_URL: str = "https://api.x.ai/v1"
    """Grok base url."""

    MOBILERUN_TELEMETRY_ENABLED: str | None = None
    """Telemetry enabled."""

    PHOENIX_URL: str = "http://0.0.0.0:6006"
    """Phoenix url."""

    PHOENIX_PROJECT_NAME: str = ""
    """Phoenix tracing project name."""

    LANGFUSE_HOST: str | None = None
    """Langfuse host."""

    LANGFUSE_BASE_URL: str | None = None
    """Langfuse base url."""

    @property
    def telemetry_enabled(self) -> bool:
        value = self.MOBILERUN_TELEMETRY_ENABLED
        return value is None or value.lower() in ("true", "1", "yes", "y")

    @property
    def langfuse_url(self) -> str | None:
        return self.LANGFUSE_BASE_URL or self.LANGFUSE_HOST


class AgentSecrets(_Environment):
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    """Langfuse public key."""

    LANGFUSE_SECRET_KEY: SecretStr | None = None
    """Langfuse secret key."""


def get_agent_settings() -> AgentSettings:
    return AgentSettings()


def get_agent_secrets() -> AgentSecrets:
    return AgentSecrets()

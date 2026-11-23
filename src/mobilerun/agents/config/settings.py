"""Agent-owned settings, read fresh at call time using exact environment field names."""

from pathlib import Path

from mobilerun.config.settings import get_settings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mobilerun.agents.config.env import OAuthCredentialPaths

DEFAULT_MANAGER_SYSTEM_PROMPT = "assets/prompts/manager/system.jinja2"
DEFAULT_STATELESS_MANAGER_SYSTEM_PROMPT = "assets/prompts/manager/stateless.jinja2"
DEFAULT_FAST_SYSTEM_PROMPT = "assets/prompts/fast_agent/system.jinja2"
DEFAULT_FAST_USER_PROMPT = "assets/prompts/fast_agent/user.jinja2"
DEFAULT_EXECUTOR_SYSTEM_PROMPT = "assets/prompts/executor/system.jinja2"
DEFAULT_APP_CARDS_DIR = "assets/app_cards"
DEFAULT_LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"
DEFAULT_MAX_STEPS = 15
DEFAULT_APP_CARDS_TIMEOUT = 2.0
DEFAULT_APP_CARDS_MAX_RETRIES = 2
DEFAULT_AFTER_SLEEP_ACTION = 1.0
DEFAULT_WAIT_FOR_STABLE_UI = 0.3
DEFAULT_GROK_CONTEXT_WINDOW = 500_000
CODEX_CONTEXT_WINDOW = 272_000
SETUP_TOKEN_EXPIRES_IN_SECONDS = 365 * 24 * 60 * 60
ANTHROPIC_OAUTH_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
GEMINI_OAUTH_REDIRECT_URI = "https://codeassist.google.com/authcode"
DEEPSEEK_CONTEXT_WINDOW = 1_048_576
OPENAI_GPT6_CONTEXT_WINDOW = 1_050_000
GROK_CONTEXT_WINDOW = 500_000
MINIMAX_DETERMINISTIC_TEMPERATURE = 0.1


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        env_file=None,
        secrets_dir=None,
        use_attribute_docstrings=True,
    )

    MOBILERUN_FINALIZE_BUDGET_SECONDS: float = 60.0
    """Finalize budget seconds."""

    MOBILERUN_FINALIZE_SCHEDULING_RESERVE_SECONDS: float = 1.0
    """Finalize scheduling reserve seconds."""

    MOBILERUN_FINALIZE_OBSERVATION_SECONDS: float = 15.0
    """Finalize observation seconds."""

    MOBILERUN_FINALIZE_TELEMETRY_SECONDS: float = 10.0
    """Finalize telemetry seconds."""

    MOBILERUN_FINALIZE_TRAJECTORY_SECONDS: float = 30.0
    """Finalize trajectory seconds."""

    MOBILERUN_FINALIZE_MCP_SECONDS: float = 3.0
    """Finalize mcp seconds."""

    MOBILERUN_FINALIZE_CLEANUP_MAX_FRACTION: float = 0.5
    """Finalize cleanup max fraction."""

    MOBILERUN_MACRO_FOCUS_SETTLE_SECONDS: float = 0.5
    """Macro focus settle seconds."""

    MOBILERUN_MAX_ERROR_CHARS: int = 2000
    """Max error chars."""

    MOBILERUN_MAX_TITLE_CHARS: int = 100
    """Max title chars."""

    MOBILERUN_TITLE_SCAN_CHARS: int = 20000
    """Title scan chars."""

    MOBILERUN_LLM_HISTORY_LIMIT: int = 100
    """Llm history limit."""

    MOBILERUN_MALFORMED_TOOL_CALL_LIMIT: int = 3
    """Malformed tool call limit."""

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

    MOBILERUN_ZAI_CODING_BASE_URL: str = "https://api.z.ai/api/coding/paas/v4"
    """Zai coding base url."""

    MOBILERUN_OLLAMA_BASE_URL: str = "http://localhost:11434"
    """Ollama base url."""

    MOBILERUN_GROK_BASE_URL: str = "https://api.x.ai/v1"
    """Grok base url."""

    MOBILERUN_MAX_IMAGE_SIZE_KB: int = 10000
    """Max image size kb."""

    MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS: float = 60.0
    """Openai oauth default timeout seconds."""

    MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_TIMEOUT_SECONDS: float = 60.0
    """Anthropic oauth default timeout seconds."""

    MOBILERUN_GEMINI_OAUTH_DEFAULT_TIMEOUT_SECONDS: float = 60.0
    """Gemini oauth default timeout seconds."""

    MOBILERUN_GROK_OAUTH_DEFAULT_TIMEOUT_SECONDS: float = 60.0
    """Grok oauth default timeout seconds."""

    MOBILERUN_OPENAI_OAUTH_DEVICE_CODE_RETRIES: int = 2
    """Openai oauth device code retries."""

    MOBILERUN_GROK_OAUTH_REQUEST_TIMEOUT_SECONDS: float = 20.0
    """Grok oauth request timeout seconds."""

    MOBILERUN_OPENAI_OAUTH_LOGIN_TIMEOUT_SECONDS: float = 300.0
    """Openai oauth login timeout seconds."""

    MOBILERUN_ANTHROPIC_OAUTH_LOGIN_TIMEOUT_SECONDS: float = 300.0
    """Anthropic oauth login timeout seconds."""

    MOBILERUN_GEMINI_OAUTH_LOGIN_TIMEOUT_SECONDS: float = 300.0
    """Gemini oauth login timeout seconds."""

    MOBILERUN_GROK_OAUTH_LOGIN_TIMEOUT_SECONDS: float = 300.0
    """Grok oauth login timeout seconds."""

    MOBILERUN_OPENAI_OAUTH_CALLBACK_PORT: int = 1455
    """Openai oauth callback port."""

    MOBILERUN_ANTHROPIC_OAUTH_CALLBACK_PORT: int = 0
    """Anthropic oauth callback port."""

    MOBILERUN_GEMINI_OAUTH_CALLBACK_PORT: int = 0
    """Gemini oauth callback port."""

    MOBILERUN_GROK_OAUTH_CALLBACK_PORT: int = 0
    """Grok oauth callback port."""

    MOBILERUN_OAUTH_DEFAULT_REFRESH_SKEW_SECONDS: float = 300.0
    """Oauth default refresh skew seconds."""

    MOBILERUN_OPENAI_OAUTH_REFRESH_SKEW_SECONDS: float = 60.0
    """Openai oauth refresh skew seconds."""

    MOBILERUN_OPENAI_OAUTH_DEVICE_CODE_TIMEOUT: int = 900
    """Openai oauth device code timeout."""

    MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_MAX_TOKENS: int = 8192
    """Anthropic oauth default max tokens."""

    MOBILERUN_ANTHROPIC_OAUTH_MAX_CODE_ATTEMPTS: int = 2
    """Anthropic oauth max code attempts."""

    MOBILERUN_GEMINI_OAUTH_MAX_CODE_ATTEMPTS: int = 2
    """Gemini oauth max code attempts."""

    MOBILERUN_ANTHROPIC_OAUTH_REFRESH_BUFFER_SECONDS: int = 300
    """Anthropic oauth refresh buffer seconds."""

    MOBILERUN_GEMINI_OAUTH_REFRESH_BUFFER_SECONDS: int = 300
    """Gemini oauth refresh buffer seconds."""

    MOBILERUN_GROK_OAUTH_TOKEN_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.25, 0.5)
    """Grok oauth token retry backoff seconds."""

    MOBILERUN_ANTHROPIC_OAUTH_API_BASE: str = "https://api.anthropic.com"
    """Anthropic oauth api base."""

    MOBILERUN_ANTHROPIC_OAUTH_TOKEN_URL: str = "https://platform.claude.com/v1/oauth/token"  # fmt: skip
    """Anthropic oauth token url."""

    MOBILERUN_ANTHROPIC_OAUTH_AUTHORIZE_URL: str = "https://platform.claude.com/oauth/authorize"  # fmt: skip
    """Anthropic oauth authorize url."""

    MOBILERUN_ANTHROPIC_OAUTH_MODERN_AUTHORIZE_URL: str = "https://claude.com/cai/oauth/authorize"  # fmt: skip
    """Anthropic oauth modern authorize url."""

    MOBILERUN_GEMINI_OAUTH_CODE_ASSIST_ENDPOINT: str = "https://daily-cloudcode-pa.googleapis.com"  # fmt: skip
    """Gemini oauth code assist endpoint."""

    MOBILERUN_GEMINI_OAUTH_TOKEN_URL: str = "https://oauth2.googleapis.com/token"
    """Gemini oauth token url."""

    MOBILERUN_GEMINI_OAUTH_AUTHORIZE_URL: str = "https://accounts.google.com/o/oauth2/v2/auth"  # fmt: skip
    """Gemini oauth authorize url."""

    MOBILERUN_OPENAI_OAUTH_ISSUER: str = "https://auth.openai.com"
    """Openai oauth issuer."""

    MOBILERUN_OPENAI_OAUTH_TOKEN_URL: str = "https://auth.openai.com/oauth/token"
    """Openai oauth token url."""

    MOBILERUN_OPENAI_OAUTH_AUTHORIZE_URL: str = "https://auth.openai.com/oauth/authorize"  # fmt: skip
    """Openai oauth authorize url."""

    MOBILERUN_OPENAI_OAUTH_API_BASE: str = "https://api.openai.com/v1"
    """Openai oauth api base."""

    MOBILERUN_OPENAI_OAUTH_CODEX_API_BASE: str = "https://chatgpt.com/backend-api/codex"
    """Openai oauth codex api base."""

    MOBILERUN_OPENAI_OAUTH_BACKEND_API_BASE: str = "https://chatgpt.com/backend-api"
    """Openai oauth backend api base."""

    MOBILERUN_GROK_OAUTH_AUTHORIZE_URL: str = "https://auth.x.ai/oauth2/authorize"
    """Grok oauth authorize url."""

    MOBILERUN_GROK_OAUTH_DEVICE_URL: str = "https://auth.x.ai/oauth2/device/code"
    """Grok oauth device url."""

    MOBILERUN_GROK_OAUTH_TOKEN_URL: str = "https://auth.x.ai/oauth2/token"
    """Grok oauth token url."""

    MOBILERUN_GROK_OAUTH_JWKS_URL: str = "https://auth.x.ai/.well-known/jwks.json"
    """Grok oauth jwks url."""

    MOBILERUN_GROK_OAUTH_PROXY: str = "https://cli-chat-proxy.grok.com/v1"
    """Grok oauth proxy."""

    MOBILERUN_TELEMETRY_ENABLED: str | None = None
    """Telemetry enabled."""

    DROIDRUN_TELEMETRY_ENABLED: str | None = None
    """Droidrun telemetry enabled."""

    MOBILERUN_STREAM_SCREENSHOTS: str | None = None
    """Stream screenshots."""

    DROIDRUN_STREAM_SCREENSHOTS: str | None = None
    """Droidrun stream screenshots."""

    MOBILERUN_OAUTH_MANUAL: str | None = None
    """Oauth manual."""

    DROIDRUN_OAUTH_MANUAL: str | None = None
    """Droidrun oauth manual."""

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
        if value is None:
            value = self.DROIDRUN_TELEMETRY_ENABLED or "true"
        return value.lower() in ("true", "1", "yes", "y")

    @property
    def stream_screenshots(self) -> bool:
        value = self.MOBILERUN_STREAM_SCREENSHOTS or self.DROIDRUN_STREAM_SCREENSHOTS or ""
        return value.lower() in ("1", "true")

    @property
    def oauth_manual(self) -> bool:
        value = self.MOBILERUN_OAUTH_MANUAL or self.DROIDRUN_OAUTH_MANUAL or ""
        return value.lower() in ("1", "true", "yes")

    @property
    def langfuse_url(self) -> str | None:
        return self.LANGFUSE_BASE_URL or self.LANGFUSE_HOST

    @property
    def oauth_credential_dir(self) -> Path:
        return self.MOBILERUN_USER_CONFIG_DIR / "credentials"

    @property
    def auth_profiles_path(self) -> Path:
        return self.oauth_credential_dir / "auth-profiles.json"

    @property
    def openai_oauth_credential_path(self) -> Path:
        return OAuthCredentialPaths().OPENAI_OAUTH_CREDENTIAL_PATH or self.auth_profiles_path

    @property
    def anthropic_oauth_credential_path(self) -> Path:
        return OAuthCredentialPaths().ANTHROPIC_OAUTH_CREDENTIAL_PATH or self.auth_profiles_path

    @property
    def gemini_oauth_credential_path(self) -> Path:
        return OAuthCredentialPaths().GEMINI_OAUTH_CREDENTIAL_PATH or self.auth_profiles_path

    @property
    def grok_oauth_credential_path(self) -> Path:
        return OAuthCredentialPaths().GROK_OAUTH_CREDENTIAL_PATH or self.auth_profiles_path

    # Shared input, captured in this snapshot rather than re-read by properties.
    MOBILERUN_USER_CONFIG_DIR: Path = Field(default_factory=lambda: get_settings().MOBILERUN_USER_CONFIG_DIR)  # fmt: skip
    """User configuration directory."""


def get_agent_settings() -> AgentSettings:
    return AgentSettings()

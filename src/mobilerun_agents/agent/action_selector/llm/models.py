"""Provider routing and selector capabilities not supplied by LlamaIndex."""

from dataclasses import dataclass, field
from typing import Any, Literal

from llama_index.core.llms import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai.responses import OpenAIResponses
from llama_index.llms.openai_like import OpenAILike, OpenAILikeResponses
from llama_index.llms.openrouter import OpenRouter

from mobilerun_agents.config.settings import get_agent_settings


# LlamaIndex retries the call; disable nested retries on its OpenAI client.
class _Responses(OpenAIResponses):
    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        return super()._get_credential_kwargs(is_async) | {"max_retries": 0}


class _Chat(OpenAILike):
    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        return super()._get_credential_kwargs(is_async) | {"max_retries": 0}


class _CompatibleResponses(OpenAILikeResponses):
    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        return super()._get_credential_kwargs(is_async) | {"max_retries": 0}


class _DeepSeek(_Chat):
    """Distinguish this native-tool provider from generic OpenAI-compatible chat."""


class _Router(OpenRouter):
    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        return super()._get_credential_kwargs(is_async) | {"max_retries": 0}


@dataclass(frozen=True)
class Capabilities:
    tool_call_format: Literal["responses", "chat_completions"] | None = None
    max_image_side: int | None = None
    normalized_coords: bool = False
    vision: bool = False
    image_token_limit: int | None = None
    grounding_max_side: int | None = None


@dataclass(frozen=True)
class ModelEntry:
    adapter: type[LLM]
    env_var: str | None = None
    capabilities: Capabilities = Capabilities()
    kwargs: dict[str, object] = field(default_factory=dict)
    vision_default: bool = False


_settings = get_agent_settings()
_responses = Capabilities(tool_call_format="responses")
MODELS: dict[str, ModelEntry] = {
    "openai": ModelEntry(_Responses, "OPENAI_API_KEY", _responses, vision_default=True),
    "anthropic": ModelEntry(
        Anthropic,
        "ANTHROPIC_API_KEY",
        Capabilities(max_image_side=1568, image_token_limit=1568),
        vision_default=True,
    ),
    "gemini": ModelEntry(GoogleGenAI, "GOOGLE_API_KEY", vision_default=True),
    "ollama": ModelEntry(
        Ollama,
        kwargs={
            "base_url": _settings.MOBILERUN_OLLAMA_BASE_URL,
            "context_window": _settings.MOBILERUN_OLLAMA_DEFAULT_CONTEXT_WINDOW,
        },
    ),
    "openrouter": ModelEntry(_Router, "OPENROUTER_API_KEY"),
    "openai_like": ModelEntry(_Chat, "OPENAI_LIKE_API_KEY", kwargs={"is_chat_model": True}),
    "deepseek": ModelEntry(
        _DeepSeek,
        "DEEPSEEK_API_KEY",
        Capabilities("chat_completions"),
        {
            "api_base": _settings.MOBILERUN_DEEPSEEK_BASE_URL,
            "is_chat_model": True,
            "is_function_calling_model": True,
        },
    ),
    "minimax": ModelEntry(
        _Chat,
        "MINIMAX_API_KEY",
        kwargs={"is_chat_model": True, "api_base": _settings.MOBILERUN_MINIMAX_BASE_URL_GLOBAL},
    ),
    "zai": ModelEntry(
        _Chat,
        "ZAI_API_KEY",
        kwargs={"is_chat_model": True, "api_base": _settings.MOBILERUN_ZAI_BASE_URL},
    ),
    "grok": ModelEntry(
        _CompatibleResponses,
        "XAI_API_KEY",
        _responses,
        {"api_base": _settings.MOBILERUN_GROK_BASE_URL},
    ),
}

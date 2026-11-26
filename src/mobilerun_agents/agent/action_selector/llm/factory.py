"""Construct role-local LLMs; no global LlamaIndex settings or client cache."""

import os
from dataclasses import replace
from typing import Any

from google.genai.types import HttpOptions, HttpRetryOptions
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.responses import OpenAIResponses
from pydantic import PositiveFloat, TypeAdapter

from mobilerun_agents.agent.auth import resolve
from mobilerun_agents.agent.logs.diagnostics import protect
from mobilerun_agents.errors import UnknownModel

from .models import MODELS as MODELS
from .models import Capabilities as Capabilities
from .models import ModelEntry as ModelEntry


def capabilities(provider: str | None, model: str) -> Capabilities:
    caps = MODELS[provider].capabilities if provider else Capabilities()
    name = model.rsplit("/", 1)[-1].lower()
    claude = provider == "anthropic" or name.startswith("claude-")
    if claude:
        caps = replace(MODELS["anthropic"].capabilities, tool_call_format=caps.tool_call_format)
    # Sonnet 5 accepts high resolution but grounding is capped at 1568.
    if claude and name in {
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5",
        "claude-sonnet-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
    }:
        caps = replace(
            caps,
            max_image_side=2576,
            image_token_limit=4784,
            grounding_max_side=1568 if name == "claude-sonnet-5" else None,
        )
    return replace(
        caps,
        normalized_coords="gemma" in name or "gemini-3.8-flash" in name,
        vision=bool(provider and MODELS[provider].vision_default),
    )


def adapter_capabilities(llm: LLM) -> Capabilities:
    """Resolve a caller-supplied adapter without asking a service for model info."""
    match llm:
        case Anthropic():
            return capabilities("anthropic", llm.model)
        case GoogleGenAI() | Ollama() | OpenAI() | OpenAIResponses():
            # In particular Ollama.metadata can make a live context-window lookup.
            model = llm.model
        case _:
            model = llm.metadata.model_name
    provider = next((key for key, entry in MODELS.items() if type(llm) is entry.adapter), None)
    if provider is None and type(llm) in (OpenAI, OpenAIResponses):
        provider = "openai"
    return capabilities(provider, model)


async def build_llm(
    provider: str, model: str, api_key: str | None = None, **options: object
) -> LLM:
    entry = MODELS[provider]
    kwargs: dict[str, Any] = {**entry.kwargs, **options, "model": model}
    if entry.env_var is not None:
        credential = await resolve(provider, api_key or os.environ.get(entry.env_var))
        kwargs["api_key"] = credential.api_key
        protect(credential.api_key)

    if provider == "gemini":
        # Supplying both limits prevents the adapter from fetching model metadata.
        kwargs.setdefault("context_window", DEFAULT_CONTEXT_WINDOW)
        kwargs.setdefault("max_tokens", DEFAULT_NUM_OUTPUTS)
        http = HttpOptions.model_validate(kwargs.pop("http_options", {})).model_copy()
        http.retry_options = HttpRetryOptions(attempts=1)
        if "timeout" in kwargs:
            seconds = TypeAdapter(PositiveFloat).validate_python(kwargs.pop("timeout"))
            http.timeout = max(1, round(seconds * 1000))
        kwargs["http_options"] = http
    elif provider == "ollama":
        if "timeout" in kwargs:
            kwargs["request_timeout"] = kwargs.pop("timeout")
        if kwargs.pop("max_retries", 0):
            raise ValueError("Ollama does not support transport retries.")

    llm = entry.adapter(**kwargs)
    # These adapters check a local model registry. Others can accept arbitrary
    # model IDs or use a live metadata lookup, so do not probe them here.
    if provider in {"openai", "anthropic"}:
        try:
            _ = llm.metadata
        except ValueError as exc:
            if not str(exc).startswith(("Unknown model ", "Unknown model:")):
                raise
            raise UnknownModel(provider, model) from None
    return llm

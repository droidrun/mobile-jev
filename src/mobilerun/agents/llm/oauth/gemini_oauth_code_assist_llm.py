import base64
import json
import secrets
import time
from collections.abc import Mapping
from typing import Any, ClassVar, Dict, Optional, Sequence

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM

from mobilerun.agents.clients.oauth.gemini import (
    _GEMINI_SAMPLING_CONFIG_KEYS as _GEMINI_SAMPLING_CONFIG_KEYS,
)
from mobilerun.agents.clients.oauth.gemini import (
    _IGNORED_REQUEST_KWARGS as _IGNORED_REQUEST_KWARGS,
)
from mobilerun.agents.clients.oauth.gemini import (
    _UNSET as _UNSET,
)
from mobilerun.agents.clients.oauth.gemini import (
    ANTIGRAVITY_CLIENT_VERSION as ANTIGRAVITY_CLIENT_VERSION,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_CLIENT_ID as DEFAULT_CLIENT_ID,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_CLIENT_SECRET as DEFAULT_CLIENT_SECRET,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_CODE_ASSIST_API_VERSION as DEFAULT_CODE_ASSIST_API_VERSION,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_CODE_ASSIST_MODELS_METHOD as DEFAULT_CODE_ASSIST_MODELS_METHOD,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_CREDENTIAL_SLOT as DEFAULT_CREDENTIAL_SLOT,
)

# Antigravity CLI installed-app OAuth client.
# Credential slot in the shared auth-profiles.json.
# Code Assist gates newer models on an Antigravity app version >= 2.5.2.
# Kwargs that must never be forwarded to Google's API (LlamaIndex-internal, and
# project ids which the consumer entitlement does not use).
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_MODEL as DEFAULT_MODEL,
)
from mobilerun.agents.clients.oauth.gemini import (
    DEFAULT_OAUTH_SCOPES as DEFAULT_OAUTH_SCOPES,
)
from mobilerun.agents.clients.oauth.gemini import (
    GeminiOAuthSession as GeminiOAuthSession,
)
from mobilerun.agents.clients.oauth.gemini import (
    _antigravity_user_agent as _antigravity_user_agent,
)
from mobilerun.agents.clients.oauth.gemini import (
    _b64_no_pad as _b64_no_pad,
)
from mobilerun.agents.clients.oauth.gemini import (
    _is_headless_environment as _is_headless_environment,
)
from mobilerun.agents.clients.oauth.gemini import (
    _normalize_manual_code as _normalize_manual_code,
)
from mobilerun.agents.clients.oauth.gemini import (
    _pkce_pair as _pkce_pair,
)
from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.llm.providers.registry import (
    GEMINI_OAUTH_DEFAULT_MODEL,
    GEMINI_OAUTH_RETIRED_MODELS,
    gemini_model_omits_sampling_params,
)


def _camel_case_keys(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Use the API's camelCase field names so snake_case overrides don't collide."""
    converted: Dict[str, Any] = {}
    for key, value in config.items():
        head, *rest = str(key).split("_")
        converted[head + "".join(part[:1].upper() + part[1:] for part in rest)] = value
    return converted


class GeminiOAuthCodeAssistLLM(CustomLLM, GeminiOAuthSession):
    """Gemini OAuth LLM that talks to Google Code Assist endpoints.

    This class expects at least one of:
    - `access_token`
    - `refresh_token`
    - `credential_path` file containing cached OAuth credentials.

    Cached credentials are stored in mobilerun's config dir.
    """

    MODEL_PRESETS: ClassVar[Dict[str, str]] = {
        "flash": GEMINI_OAUTH_DEFAULT_MODEL,
        "pro": "gemini-pro-agent",
        "flash_lite": "gemini-3.5-flash-lite",
    }

    model: str = Field(default=DEFAULT_MODEL, description="Gemini model id.")
    model_preset: Optional[str] = Field(
        default=None,
        description="Quick model selector key from MODEL_PRESETS.",
    )
    custom_model: Optional[str] = Field(
        default=None,
        description="Optional custom model id; overrides model/model_preset.",
    )
    credential_slot: str = Field(
        default=DEFAULT_CREDENTIAL_SLOT,
        description="Nested key in the shared auth-profiles.json credential file.",
    )
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    timeout: float = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_GEMINI_OAUTH_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
    )

    access_token: Optional[str] = Field(default=None, description="OAuth access token.")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token.")
    client_id: str = Field(default=DEFAULT_CLIENT_ID)
    client_secret: str = Field(default=DEFAULT_CLIENT_SECRET)
    authorize_url: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_GEMINI_OAUTH_AUTHORIZE_URL
    )
    token_url: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_GEMINI_OAUTH_TOKEN_URL
    )
    refresh_buffer_seconds: int = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_GEMINI_OAUTH_REFRESH_BUFFER_SECONDS,
        ge=0,
    )

    code_assist_endpoint: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_GEMINI_OAUTH_CODE_ASSIST_ENDPOINT
    )
    code_assist_api_version: str = Field(default=DEFAULT_CODE_ASSIST_API_VERSION)
    scopes: tuple = Field(default=DEFAULT_OAUTH_SCOPES)

    credential_path: Optional[str] = Field(
        default_factory=lambda: str(get_agent_settings().gemini_oauth_credential_path),
        description="Optional path to JSON OAuth credentials cache.",
    )
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _session: requests.Session = PrivateAttr()
    _cached_access_token: Optional[str] = PrivateAttr(default=None)
    _cached_refresh_token: Optional[str] = PrivateAttr(default=None)
    _access_token_expiry: Optional[float] = PrivateAttr(default=None)

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        model_preset: Optional[str] = None,
        custom_model: Optional[str] = None,
        credential_slot: str = DEFAULT_CREDENTIAL_SLOT,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float | None = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        client_id: str = DEFAULT_CLIENT_ID,
        client_secret: str = DEFAULT_CLIENT_SECRET,
        authorize_url: str | None = None,
        token_url: str | None = None,
        refresh_buffer_seconds: int | None = None,
        code_assist_endpoint: str | None = None,
        code_assist_api_version: str = DEFAULT_CODE_ASSIST_API_VERSION,
        scopes: Optional[Sequence[str]] = None,
        credential_path: Optional[str] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        if timeout is None:
            timeout = get_agent_settings().MOBILERUN_GEMINI_OAUTH_DEFAULT_TIMEOUT_SECONDS
        if refresh_buffer_seconds is None:
            refresh_buffer_seconds = (
                get_agent_settings().MOBILERUN_GEMINI_OAUTH_REFRESH_BUFFER_SECONDS
            )
        if authorize_url is None:
            authorize_url = get_agent_settings().MOBILERUN_GEMINI_OAUTH_AUTHORIZE_URL
        if token_url is None:
            token_url = get_agent_settings().MOBILERUN_GEMINI_OAUTH_TOKEN_URL
        if code_assist_endpoint is None:
            code_assist_endpoint = get_agent_settings().MOBILERUN_GEMINI_OAUTH_CODE_ASSIST_ENDPOINT
        if credential_path is None:
            credential_path = str(get_agent_settings().gemini_oauth_credential_path)
        # custom_model wins; then an explicit model (honored verbatim, even if it
        # equals DEFAULT_MODEL); then a preset key; else the default model.
        selected_model = custom_model
        if not selected_model:
            if model in self.MODEL_PRESETS:
                selected_model = self.MODEL_PRESETS[model]
            elif model:
                selected_model = model
            elif model_preset in self.MODEL_PRESETS:
                selected_model = self.MODEL_PRESETS[model_preset]
            else:
                selected_model = DEFAULT_MODEL

        if selected_model in GEMINI_OAUTH_RETIRED_MODELS:
            raise ValueError(
                f"Gemini OAuth model '{selected_model}' is retired. "
                f"Use '{GEMINI_OAUTH_DEFAULT_MODEL}' or re-run "
                "`mobilerun agent llm configure` to choose an available model."
            )

        super().__init__(
            model=selected_model,
            model_preset=model_preset,
            custom_model=custom_model,
            credential_slot=credential_slot,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
            authorize_url=authorize_url,
            token_url=token_url,
            refresh_buffer_seconds=refresh_buffer_seconds,
            code_assist_endpoint=code_assist_endpoint,
            code_assist_api_version=code_assist_api_version,
            scopes=tuple(scopes) if scopes else DEFAULT_OAUTH_SCOPES,
            credential_path=credential_path,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
        )

        self._session = requests.Session()
        self._cached_access_token = access_token
        self._cached_refresh_token = refresh_token

        if credential_path:
            self._load_credentials_from_file(credential_path)

    @classmethod
    def available_model_presets(cls) -> Dict[str, str]:
        return dict(cls.MODEL_PRESETS)

    @classmethod
    def class_name(cls) -> str:
        return "gemini_oauth_code_assist"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=1_000_000,
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return ""

    @staticmethod
    def _image_mime_type(raw: bytes) -> str:
        if raw.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw.startswith(b"GIF8"):
            return "image/gif"
        return "image/png"

    @classmethod
    def _message_parts(cls, message: ChatMessage) -> list[Dict[str, Any]]:
        parts: list[Dict[str, Any]] = []
        for block in message.blocks or []:
            if isinstance(block, TextBlock):
                if block.text:
                    parts.append({"text": block.text})
            elif isinstance(block, ImageBlock):
                raw = block.resolve_image(as_base64=False).read()
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": cls._image_mime_type(raw),
                            "data": base64.b64encode(raw).decode("ascii"),
                        }
                    }
                )
        return parts

    def _to_code_assist_request(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # NOTE: deliberately not using ``self.convert_chat_messages`` — the
        # llama_index base helper is text-only and raises on ImageBlocks.
        contents: list[Dict[str, Any]] = []
        system_chunks: list[str] = []

        for msg in messages:
            parts = self._message_parts(msg)
            if msg.role == MessageRole.SYSTEM:
                system_chunks.extend(part["text"] for part in parts if part.get("text"))
                continue

            role = "model" if msg.role == MessageRole.ASSISTANT else "user"
            if any("inlineData" in part for part in parts):
                contents.append({"role": role, "parts": parts})
            else:
                # Text-only messages keep the single-text-part shape the API
                # already receives.
                contents.append({"role": role, "parts": [{"text": self._message_text(msg)}]})

        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        generation_config: Dict[str, Any] = {}
        if not gemini_model_omits_sampling_params(self.model):
            generation_config["temperature"] = self.temperature
        if self.max_tokens is not None:
            generation_config["maxOutputTokens"] = self.max_tokens
        generation_config.update(_camel_case_keys(kwargs.pop("generation_config", {})))

        request_extra = dict(self.additional_kwargs)
        call_extra = dict(kwargs.pop("request_extra", {}))
        # Merge generationConfig overrides so they don't replace the built config;
        # a non-mapping value in the highest layer still replaces it.
        replacement: Any = _UNSET
        for extra in (request_extra, call_extra):
            for key in ("generation_config", "generationConfig"):
                if key not in extra:
                    continue
                value = extra.pop(key)
                if isinstance(value, Mapping):
                    generation_config.update(_camel_case_keys(value))
                    replacement = _UNSET
                else:
                    replacement = value
        request_extra.update(call_extra)

        omit_sampling = gemini_model_omits_sampling_params(self.model)
        if omit_sampling:
            for param in _GEMINI_SAMPLING_CONFIG_KEYS:
                generation_config.pop(param, None)

        request: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": (generation_config if replacement is _UNSET else replacement),
        }

        if system_chunks:
            request["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": "\n\n".join(system_chunks)}],
            }

        for ignored in _IGNORED_REQUEST_KWARGS:
            request_extra.pop(ignored, None)
        request.update(request_extra)

        payload: Dict[str, Any] = {
            "model": self.model,
            "request": request,
            "userAgent": "droidrun",
            "requestId": f"droidrun-{int(time.time() * 1000)}-{secrets.token_hex(4)}",
        }

        # Strip internal kwargs Google rejects; the consumer entitlement is
        # project-less, so never send a project (also filtered by the ignore set).
        safe_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in _IGNORED_REQUEST_KWARGS
            and not (omit_sampling and k in _GEMINI_SAMPLING_CONFIG_KEYS)
        }
        payload.update(safe_kwargs)
        payload.pop("project", None)
        return payload

    @staticmethod
    def _extract_text(chunk: Dict[str, Any]) -> str:
        response = chunk.get("response") or {}
        candidates = response.get("candidates") or []
        if not candidates:
            return ""

        first = candidates[0] or {}
        content = first.get("content") or {}
        parts = content.get("parts") or []

        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

        return "".join(text_parts)

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        # Google's Code Assist private API (v1internal) does not reliably
        # accept the non-streaming generateContent endpoint for every model.
        # Route chat through streamGenerateContent and accumulate the stream,
        # matching the pattern used by gemini-cli / OpenClaw.
        token = self._resolve_access_token()
        payload = self._to_code_assist_request(messages, **kwargs)

        response = self._session.post(
            self._method_url("streamGenerateContent"),
            params={"alt": "sse"},
            headers={
                **self._build_headers(token),
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"Code Assist {response.status_code} error: {response.text}",
                response=response,
            )

        accumulated = ""
        last_raw: Dict[str, Any] = {}
        buffer: list[str] = []

        def _flush(buffer: list[str]) -> Optional[Dict[str, Any]]:
            if not buffer:
                return None
            chunk_text = "\n".join(buffer)
            try:
                return json.loads(chunk_text)
            except json.JSONDecodeError:
                return None

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            stripped = line.strip()
            if stripped.startswith("data:"):
                buffer.append(stripped[5:].strip())
                continue
            if stripped != "" or not buffer:
                continue
            raw_chunk = _flush(buffer)
            buffer = []
            if raw_chunk is None:
                continue
            last_raw = raw_chunk
            delta = self._extract_text(raw_chunk)
            if delta:
                accumulated += delta

        raw_chunk = _flush(buffer)
        if raw_chunk is not None:
            last_raw = raw_chunk
            delta = self._extract_text(raw_chunk)
            if delta:
                accumulated += delta

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated),
            raw=last_raw,
            additional_kwargs={
                "trace_id": last_raw.get("traceId"),
                "usage": (last_raw.get("response") or {}).get("usageMetadata"),
                "model_version": (last_raw.get("response") or {}).get("modelVersion"),
            },
        )

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        del formatted  # The Code Assist request already ignores this LlamaIndex flag.
        chat_response = self.chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            **kwargs,
        )
        return CompletionResponse(
            text=chat_response.message.content or "",
            raw=chat_response.raw,
            additional_kwargs=chat_response.additional_kwargs,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        token = self._resolve_access_token()
        payload = self._to_code_assist_request(messages, **kwargs)

        response = self._session.post(
            self._method_url("streamGenerateContent"),
            params={"alt": "sse"},
            headers={
                **self._build_headers(token),
                "Accept": "text/event-stream",
            },
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        def gen() -> ChatResponseGen:
            accumulated = ""
            buffer: list[str] = []

            for line in response.iter_lines(decode_unicode=True):
                if line is None:
                    continue

                stripped = line.strip()
                if stripped.startswith("data:"):
                    buffer.append(stripped[5:].strip())
                    continue

                if stripped != "" or not buffer:
                    continue

                chunk_text = "\n".join(buffer)
                buffer = []
                try:
                    raw_chunk = json.loads(chunk_text)
                except json.JSONDecodeError:
                    continue

                delta = self._extract_text(raw_chunk)
                if not delta:
                    continue

                accumulated += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=accumulated),
                    delta=delta,
                    raw=raw_chunk,
                    additional_kwargs={
                        "trace_id": raw_chunk.get("traceId"),
                        "usage": (raw_chunk.get("response") or {}).get("usageMetadata"),
                        "model_version": (raw_chunk.get("response") or {}).get("modelVersion"),
                    },
                )

            if buffer:
                chunk_text = "\n".join(buffer)
                try:
                    raw_chunk = json.loads(chunk_text)
                except json.JSONDecodeError:
                    raw_chunk = None

                if raw_chunk:
                    delta = self._extract_text(raw_chunk)
                    if delta:
                        accumulated += delta
                        yield ChatResponse(
                            message=ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=accumulated,
                            ),
                            delta=delta,
                            raw=raw_chunk,
                            additional_kwargs={
                                "trace_id": raw_chunk.get("traceId"),
                                "usage": (raw_chunk.get("response") or {}).get("usageMetadata"),
                                "model_version": (raw_chunk.get("response") or {}).get(
                                    "modelVersion"
                                ),
                            },
                        )

        return gen()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        del formatted  # The Code Assist request already ignores this LlamaIndex flag.

        def gen() -> CompletionResponseGen:
            for chat_chunk in self.stream_chat(
                [ChatMessage(role=MessageRole.USER, content=prompt)],
                **kwargs,
            ):
                yield CompletionResponse(
                    text=chat_chunk.message.content or "",
                    delta=chat_chunk.delta,
                    raw=chat_chunk.raw,
                    additional_kwargs=chat_chunk.additional_kwargs,
                )

        return gen()

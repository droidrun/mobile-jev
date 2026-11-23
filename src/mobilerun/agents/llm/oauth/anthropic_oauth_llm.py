import base64
import contextlib
import json
import weakref
from typing import Any, Dict, Iterator, Literal, Optional, Sequence

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
from mobilerun.errors.base import AgentError

from mobilerun.agents.clients.oauth.anthropic import (
    _ERROR_TYPE_STATUS as _ERROR_TYPE_STATUS,
)
from mobilerun.agents.clients.oauth.anthropic import (
    _IGNORED_REQUEST_KWARGS as _IGNORED_REQUEST_KWARGS,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_ANTHROPIC_VERSION as DEFAULT_ANTHROPIC_VERSION,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_CC_ENTRYPOINT as DEFAULT_CC_ENTRYPOINT,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_CC_VERSION as DEFAULT_CC_VERSION,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_CLAUDE_CODE_VERSION as DEFAULT_CLAUDE_CODE_VERSION,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_CLIENT_ID as DEFAULT_CLIENT_ID,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_LOGIN_SCOPE as DEFAULT_LOGIN_SCOPE,
)

# Model access is Claude Code version-gated, so keep the HTTP user agent and
# billing marker derived from the same verified-compatible client version.
# Socket read timeout. chat() reads nothing until the reply is complete, and
# always-on thinking makes that slower.
# HTTP status equivalent of each error type, so stream errors retry like HTTP ones.
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_MODEL as DEFAULT_MODEL,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_MODERN_LOGIN_SCOPE as DEFAULT_MODERN_LOGIN_SCOPE,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_OAUTH_BETA as DEFAULT_OAUTH_BETA,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_REFRESH_SCOPE as DEFAULT_REFRESH_SCOPE,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_SETUP_TOKEN_SCOPE as DEFAULT_SETUP_TOKEN_SCOPE,
)
from mobilerun.agents.clients.oauth.anthropic import (
    DEFAULT_USER_AGENT as DEFAULT_USER_AGENT,
)
from mobilerun.agents.clients.oauth.anthropic import (
    AnthropicOAuthSession as AnthropicOAuthSession,
)
from mobilerun.agents.clients.oauth.anthropic import (
    _b64_no_pad as _b64_no_pad,
)
from mobilerun.agents.clients.oauth.anthropic import (
    _is_headless_environment as _is_headless_environment,
)
from mobilerun.agents.clients.oauth.anthropic import (
    _normalize_manual_code as _normalize_manual_code,
)
from mobilerun.agents.clients.oauth.anthropic import (
    _pkce_pair as _pkce_pair,
)
from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.llm.providers.anthropic import (
    anthropic_model_context_window,
    anthropic_model_omits_sampling_params,
    strip_anthropic_sampling_params,
)


class AnthropicOAuthStreamError(AgentError, requests.HTTPError):
    """Error event in, or early end of, a streamed Messages response."""

    def __init__(
        self,
        message: str,
        *,
        error_type: Optional[str] = None,
        request: Any = None,
    ) -> None:
        super().__init__(message, request=request)
        self.error_type = error_type
        self.status_code = _ERROR_TYPE_STATUS.get(error_type or "")


def _iter_sse_events(response: Any) -> Iterator[Dict[str, Any]]:
    """Yield the JSON data of each server-sent event."""
    event_name: Optional[str] = None
    data_lines: list[str] = []

    def dispatch() -> Optional[Dict[str, Any]]:
        if not data_lines:
            return None
        event = json.loads("\n".join(data_lines))
        if isinstance(event, dict) and event_name and "type" not in event:
            event["type"] = event_name
        return event

    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line:
            event = dispatch()
            if event is not None:
                yield event
            event_name, data_lines = None, []
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    event = dispatch()
    if event is not None:
        yield event


class AnthropicOAuthLLM(CustomLLM, AnthropicOAuthSession):
    """Anthropic OAuth-backed LLM using Claude subscription OAuth tokens."""

    model: str = Field(default=DEFAULT_MODEL, description="Anthropic model id.")
    max_tokens: Optional[int] = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_MAX_TOKENS,
        gt=0,
    )
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    timeout: float = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
    )

    access_token: Optional[str] = Field(default=None, description="OAuth access token.")
    refresh_token: Optional[str] = Field(default=None, description="OAuth refresh token.")
    client_id: str = Field(default=DEFAULT_CLIENT_ID)
    authorize_url: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_AUTHORIZE_URL
    )
    token_url: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_TOKEN_URL
    )
    login_scope: str = Field(default=DEFAULT_LOGIN_SCOPE)
    refresh_scope: str = Field(default=DEFAULT_REFRESH_SCOPE)
    refresh_buffer_seconds: int = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_REFRESH_BUFFER_SECONDS,
        ge=0,
    )
    credential_path: Optional[str] = Field(
        default_factory=lambda: str(get_agent_settings().anthropic_oauth_credential_path)
    )

    api_base: str = Field(
        default_factory=lambda: get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_API_BASE
    )
    anthropic_version: str = Field(default=DEFAULT_ANTHROPIC_VERSION)
    oauth_beta: str = Field(default=DEFAULT_OAUTH_BETA)
    user_agent: Optional[str] = Field(default=DEFAULT_USER_AGENT)

    billing_header_mode: Literal["auto", "always", "never"] = Field(default="auto")
    cc_version: str = Field(default=DEFAULT_CC_VERSION)
    cc_entrypoint: str = Field(default=DEFAULT_CC_ENTRYPOINT)
    cch_value: str = Field(default="00000")
    inject_identifier: bool = Field(default=False)
    identifier_text: str = Field(
        default="You are Claude Code, Anthropic's official CLI for Claude."
    )

    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)

    _session: requests.Session = PrivateAttr()
    _cached_access_token: Optional[str] = PrivateAttr(default=None)
    _cached_refresh_token: Optional[str] = PrivateAttr(default=None)
    _access_token_expiry: Optional[float] = PrivateAttr(default=None)

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: Optional[int] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float | None = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        client_id: str = DEFAULT_CLIENT_ID,
        authorize_url: str | None = None,
        token_url: str | None = None,
        login_scope: str = DEFAULT_LOGIN_SCOPE,
        refresh_scope: str = DEFAULT_REFRESH_SCOPE,
        refresh_buffer_seconds: int | None = None,
        credential_path: Optional[str] = None,
        api_base: str | None = None,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        oauth_beta: str = DEFAULT_OAUTH_BETA,
        user_agent: Optional[str] = DEFAULT_USER_AGENT,
        billing_header_mode: Literal["auto", "always", "never"] = "auto",
        cc_version: str = DEFAULT_CC_VERSION,
        cc_entrypoint: str = DEFAULT_CC_ENTRYPOINT,
        cch_value: str = "00000",
        inject_identifier: bool = False,
        identifier_text: str = "You are Claude Code, Anthropic's official CLI for Claude.",
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        if max_tokens is None:
            max_tokens = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_MAX_TOKENS
        if timeout is None:
            timeout = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_TIMEOUT_SECONDS
        if refresh_buffer_seconds is None:
            refresh_buffer_seconds = (
                get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_REFRESH_BUFFER_SECONDS
            )
        if authorize_url is None:
            authorize_url = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_AUTHORIZE_URL
        if token_url is None:
            token_url = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_TOKEN_URL
        if api_base is None:
            api_base = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_API_BASE
        if credential_path is None:
            credential_path = str(get_agent_settings().anthropic_oauth_credential_path)
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=client_id,
            authorize_url=authorize_url,
            token_url=token_url,
            login_scope=login_scope,
            refresh_scope=refresh_scope,
            refresh_buffer_seconds=refresh_buffer_seconds,
            credential_path=credential_path,
            api_base=api_base,
            anthropic_version=anthropic_version,
            oauth_beta=oauth_beta,
            user_agent=user_agent,
            billing_header_mode=billing_header_mode,
            cc_version=cc_version,
            cc_entrypoint=cc_entrypoint,
            cch_value=cch_value,
            inject_identifier=inject_identifier,
            identifier_text=identifier_text,
            additional_kwargs=additional_kwargs or {},
            callback_manager=callback_manager or CallbackManager([]),
        )
        self._session = requests.Session()
        self._cached_access_token = access_token
        self._cached_refresh_token = refresh_token
        if credential_path:
            self._load_credentials_from_file(credential_path)

    @classmethod
    def class_name(cls) -> str:
        return "AnthropicOAuthLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=anthropic_model_context_window(self.model) or 200_000,
            num_output=self.max_tokens or -1,
            model_name=self.model,
            is_chat_model=True,
            # This text/image adapter does not implement native function calls.
            # Structured extraction must use the validated text program.
            is_function_calling_model=False,
        )

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            return ""
        texts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                texts.append(str(block["text"]))
        return "\n".join(texts)

    @staticmethod
    def _image_media_type(raw: bytes) -> str:
        if raw.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw.startswith(b"GIF8"):
            return "image/gif"
        return "image/png"

    @classmethod
    def _content_blocks(cls, message: ChatMessage) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in message.blocks or []:
            if isinstance(block, TextBlock):
                if block.text:
                    blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                raw = block.resolve_image(as_base64=False).read()
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": cls._image_media_type(raw),
                            "data": base64.b64encode(raw).decode("ascii"),
                        },
                    }
                )
        return blocks

    def _to_provider_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        provider_messages: list[dict[str, Any]] = []
        system_lines: list[str] = []

        for message in messages:
            role = message.role.value
            blocks = self._content_blocks(message)
            if role == MessageRole.SYSTEM.value:
                system_lines.extend(block["text"] for block in blocks if block["type"] == "text")
                continue
            if role not in {MessageRole.USER.value, MessageRole.ASSISTANT.value}:
                role = MessageRole.USER.value
            if any(block["type"] == "image" for block in blocks):
                provider_messages.append({"role": role, "content": blocks})
            else:
                # Text-only messages keep the plain-string form the API also
                # accepts, preserving existing request shapes.
                content = message.content if isinstance(message.content, str) else ""
                provider_messages.append({"role": role, "content": content})

        if not provider_messages:
            provider_messages.append({"role": MessageRole.USER.value, "content": ""})
        return provider_messages, system_lines

    def _billing_header_text(self) -> str:
        return (
            "x-anthropic-billing-header: "
            f"cc_version={self.cc_version}; "
            f"cc_entrypoint={self.cc_entrypoint}; "
            f"cch={self.cch_value};"
        )

    def _use_billing_header(self, model_id: str) -> bool:
        if self.billing_header_mode == "always":
            return True
        if self.billing_header_mode == "never":
            return False
        return not model_id.startswith("claude-haiku")

    @staticmethod
    def _supports_temperature(model_id: str) -> bool:
        """Whether the model accepts Anthropic sampling controls."""

        return not anthropic_model_omits_sampling_params(model_id)

    def _system_blocks(
        self, user_system_lines: Sequence[str], model_id: str
    ) -> list[dict[str, str]]:
        blocks: list[dict[str, str]] = []
        if self._use_billing_header(model_id):
            blocks.append({"type": "text", "text": self._billing_header_text()})
        if self.inject_identifier and self.identifier_text:
            blocks.append({"type": "text", "text": self.identifier_text})
        for line in user_system_lines:
            if line:
                blocks.append({"type": "text", "text": line})
        return blocks

    @staticmethod
    def _sanitize_request_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop framework-only kwargs that are invalid for Anthropic's API."""
        return {
            key: value
            for key, value in kwargs.items()
            if key not in _IGNORED_REQUEST_KWARGS and value is not None
        }

    def _messages_request(
        self, messages: Sequence[ChatMessage], kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Build the Messages API URL, headers and payload."""
        provider_messages, system_lines = self._to_provider_messages(messages)
        token = self._resolve_access_token()
        request_kwargs = self._sanitize_request_kwargs(kwargs)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": provider_messages,
        }
        if self._supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        system_blocks = self._system_blocks(system_lines, self.model)
        if system_blocks:
            payload["system"] = system_blocks

        payload.update(self.additional_kwargs)
        payload.update(request_kwargs)
        strip_anthropic_sampling_params(payload)

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "anthropic-beta": self.oauth_beta,
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent

        return f"{self.api_base.rstrip('/')}/v1/messages", headers, payload

    @classmethod
    def _chat_response(cls, data: Dict[str, Any], delta: Optional[str] = None) -> ChatResponse:
        assistant = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=cls._extract_text(data),
            additional_kwargs={"content_blocks": data.get("content", [])},
        )
        return ChatResponse(
            message=assistant,
            raw=data,
            delta=delta,
            additional_kwargs={
                "id": data.get("id"),
                "usage": data.get("usage"),
                "stop_reason": data.get("stop_reason"),
            },
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        url, headers, payload = self._messages_request(messages, kwargs)
        res = self._session.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        res.raise_for_status()
        return self._chat_response(res.json())

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        del formatted
        resp = self.chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            **self._sanitize_request_kwargs(kwargs),
        )
        return CompletionResponse(
            text=resp.message.content or "",
            raw=resp.raw,
            additional_kwargs=resp.additional_kwargs,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        url, headers, payload = self._messages_request(messages, kwargs)
        payload["stream"] = True
        res = self._session.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        try:
            res.raise_for_status()
        except requests.HTTPError:
            # Buffer the error body so it stays readable after closing.
            with contextlib.suppress(requests.RequestException):
                _ = res.content
            res.close()
            raise
        stream = self._stream_responses(res)
        # Close the response even if the stream is dropped unread.
        weakref.finalize(stream, res.close)
        return stream

    def _stream_responses(self, res: requests.Response) -> ChatResponseGen:
        message: Dict[str, Any] = {}
        blocks: Dict[int, Dict[str, Any]] = {}
        tool_input_json: Dict[int, str] = {}
        text = ""
        try:
            for event in _iter_sse_events(res):
                kind = event.get("type")
                if kind == "message_start":
                    message = dict(event.get("message") or {})
                    message["usage"] = dict(message.get("usage") or {})
                elif kind == "content_block_start":
                    blocks[event["index"]] = dict(event.get("content_block") or {})
                elif kind == "content_block_delta":
                    index = event["index"]
                    block = blocks[index]
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta" and delta.get("text"):
                        # Text blocks are joined with newlines, as in chat().
                        piece = delta["text"]
                        if text and not block.get("text"):
                            piece = "\n" + piece
                        block["text"] = block.get("text", "") + delta["text"]
                        text += piece
                        yield ChatResponse(
                            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                            delta=piece,
                            raw=event,
                        )
                    elif delta_type == "thinking_delta":
                        block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                    elif delta_type == "signature_delta":
                        block["signature"] = delta.get("signature", "")
                    elif delta_type == "input_json_delta":
                        tool_input_json[index] = tool_input_json.get(index, "") + delta.get(
                            "partial_json", ""
                        )
                    elif delta_type == "citations_delta":
                        block.setdefault("citations", []).append(delta.get("citation"))
                elif kind == "content_block_stop":
                    partial_json = tool_input_json.pop(event["index"], "")
                    if partial_json:
                        blocks[event["index"]]["input"] = json.loads(partial_json)
                elif kind == "message_delta":
                    message.update(event.get("delta") or {})
                    usage = message.setdefault("usage", {})
                    for key, value in (event.get("usage") or {}).items():
                        if value is not None:
                            usage[key] = value
                elif kind == "message_stop":
                    message["content"] = [blocks[index] for index in sorted(blocks)]
                    yield self._chat_response(message, delta="")
                    return
                elif kind == "error":
                    error = event.get("error") or {}
                    error_type = error.get("type")
                    raise AnthropicOAuthStreamError(
                        f"Anthropic stream error {error_type}: {error.get('message')}",
                        error_type=error_type,
                        request=res.request,
                    )
            raise AnthropicOAuthStreamError(
                "Anthropic stream ended before message_stop.",
                request=res.request,
            )
        finally:
            res.close()

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        del formatted
        chat_stream = self.stream_chat(
            [ChatMessage(role=MessageRole.USER, content=prompt)],
            **self._sanitize_request_kwargs(kwargs),
        )

        def gen() -> CompletionResponseGen:
            for chunk in chat_stream:
                yield CompletionResponse(
                    text=chunk.message.content or "",
                    delta=chunk.delta,
                    raw=chunk.raw,
                    additional_kwargs=chunk.additional_kwargs,
                )

        return gen()

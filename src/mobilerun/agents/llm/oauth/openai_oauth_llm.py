"""Standalone OpenAI OAuth subclass for LlamaIndex.

Usage:
    from openai_oauth_llm import OpenAIOAuth

    llm = OpenAIOAuth(
        auth_model="openai-codex/gpt-6-astra",
        custom_model="gpt-6.1-sol",  # optional override
        oauth_refresh_token="rt_...",
        oauth_access_token="eyJ...",  # optional if cached file already exists
        oauth_credential_path="/optional/custom/auth-profiles.json",
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, Dict, Literal, Optional, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.types import PydanticProgramMode
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import llm_retry_decorator
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncOpenAI, AsyncStream, Stream
from openai import OpenAI as OpenAIClient
from openai.lib._pydantic import to_strict_json_schema
from openai.types.responses import Response, ResponseOutputItem, ResponseStreamEvent
from openai.types.responses.response_input_param import ResponseInputParam

from mobilerun.agents.clients.oauth.openai import (
    _CODEX_CONTEXT_WINDOW as _CODEX_CONTEXT_WINDOW,
)
from mobilerun.agents.clients.oauth.openai import (
    _GPT_6_ASTRA_MODEL as _GPT_6_ASTRA_MODEL,
)
from mobilerun.agents.clients.oauth.openai import (
    _GPT_6_UNSUPPORTED_INCLUDE as _GPT_6_UNSUPPORTED_INCLUDE,
)
from mobilerun.agents.clients.oauth.openai import (
    _GPT_6_UNSUPPORTED_PARAMS as _GPT_6_UNSUPPORTED_PARAMS,
)
from mobilerun.agents.clients.oauth.openai import (
    _OPENAI_LOGIN_TIMEOUT_MESSAGE as _OPENAI_LOGIN_TIMEOUT_MESSAGE,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_AUTH_MODEL as DEFAULT_AUTH_MODEL,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_BACKEND_API_BASE as DEFAULT_BACKEND_API_BASE,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_CODEX_API_BASE as DEFAULT_CODEX_API_BASE,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_API_BASE as DEFAULT_OPENAI_API_BASE,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST as DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_CALLBACK_PATH as DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_CLIENT_ID as DEFAULT_OPENAI_OAUTH_CLIENT_ID,
)

# Codex reports this context window for ChatGPT-account models.
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_ISSUER as DEFAULT_OPENAI_OAUTH_ISSUER,
)
from mobilerun.agents.clients.oauth.openai import (
    DEFAULT_OPENAI_OAUTH_SCOPE as DEFAULT_OPENAI_OAUTH_SCOPE,
)
from mobilerun.agents.clients.oauth.openai import (
    OpenAIOAuthCredentials as OpenAIOAuthCredentials,
)
from mobilerun.agents.clients.oauth.openai import (
    OpenAIOAuthCredentialStore as OpenAIOAuthCredentialStore,
)
from mobilerun.agents.clients.oauth.openai import (
    OpenAIOAuthSession as OpenAIOAuthSession,
)
from mobilerun.agents.clients.oauth.openai import (
    OpenAIOAuthSessionManager as OpenAIOAuthSessionManager,
)
from mobilerun.agents.clients.oauth.openai import (
    _b64_no_pad as _b64_no_pad,
)
from mobilerun.agents.clients.oauth.openai import (
    _is_headless_environment as _is_headless_environment,
)
from mobilerun.agents.clients.oauth.openai import (
    _pkce_pair as _pkce_pair,
)
from mobilerun.agents.clients.oauth.openai import (
    _poll_device_code as _poll_device_code,
)
from mobilerun.agents.clients.oauth.openai import (
    _request_device_code as _request_device_code,
)
from mobilerun.agents.clients.oauth.openai import (
    _tls_preflight as _tls_preflight,
)
from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.llm.providers.registry import (
    OPENAI_ASTRA_DEFAULT_REASONING_EFFORT,
    OPENAI_GPT6_MODELS,
    OPENAI_OAUTH_DEFAULT_MODEL,
    OPENAI_OAUTH_UNSUPPORTED_MODELS,
    OPENAI_REASONING_EFFORTS,
    normalize_model_id_for_variant,
)


class OpenAIOAuth(OpenAI, OpenAIOAuthSession):
    """OpenAI LLM backed by ChatGPT OAuth refresh/access tokens."""

    supports_native_tool_calls: ClassVar[bool] = True
    _oauth_refresh_skew_ms: int
    _use_chatgpt_account_header: bool
    _oauth_account_id: str | None
    _active_access_token: str | None
    _responses_api_base: str

    # LlamaIndex OpenAI.reasoning_effort omits 'max'; the Codex OAuth API accepts it.
    reasoning_effort: Optional[  # pyright: ignore[reportIncompatibleVariableOverride]
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    ] = None

    @classmethod
    def class_name(cls) -> str:
        return "OpenAIOAuth"

    @property
    def metadata(self) -> LLMMetadata:
        # Codex model IDs are not always in llama-index's static OpenAI map.
        return LLMMetadata(
            context_window=_CODEX_CONTEXT_WINDOW,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )

    @staticmethod
    def _resolve_model_name(
        *,
        auth_model: str,
        custom_model: Optional[str],
        model: Optional[str],
    ) -> str:
        if custom_model and custom_model.strip():
            selected_model = custom_model.strip()
        elif model and model.strip():
            selected_model = model.strip()
        else:
            selected_model = auth_model.strip() or OPENAI_OAUTH_DEFAULT_MODEL

        normalized_model = normalize_model_id_for_variant(
            "openai",
            "oauth",
            selected_model,
        )
        if normalized_model != selected_model:
            return normalized_model

        # Preserve the historical auth_model behavior for custom Codex aliases
        # that are not part of the advertised catalog. Explicit model and
        # custom_model values remain untouched.
        if not (custom_model and custom_model.strip()) and not (model and model.strip()):
            if "/" in selected_model:
                return selected_model.split("/", 1)[1].strip()

        return selected_model

    def __init__(
        self,
        auth_model: str = DEFAULT_AUTH_MODEL,
        custom_model: Optional[str] = None,
        model: Optional[str] = None,
        oauth_refresh_token: Optional[str] = None,
        oauth_access_token: Optional[str] = None,
        oauth_expires_at_ms: Optional[int] = None,
        oauth_account_id: Optional[str] = None,
        oauth_credential_path: Optional[str] = None,
        oauth_client_id: str = DEFAULT_OPENAI_OAUTH_CLIENT_ID,
        oauth_issuer: str = DEFAULT_OPENAI_OAUTH_ISSUER,
        oauth_refresh_skew_seconds: int | None = None,
        use_chatgpt_account_header: bool = False,
        **kwargs: Any,
    ) -> None:
        resolved_model = self._resolve_model_name(
            auth_model=auth_model,
            custom_model=custom_model,
            model=model,
        )
        if resolved_model in OPENAI_OAUTH_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model '{resolved_model}' is not supported with OpenAI OAuth "
                f"ChatGPT-account credentials. Use '{OPENAI_OAUTH_DEFAULT_MODEL}' "
                "or re-run `mobilerun agent llm login openai`."
            )

        seed_api_key = oauth_access_token or kwargs.pop("api_key", None) or "oauth"
        if oauth_refresh_skew_seconds is None:
            oauth_refresh_skew_seconds = int(
                get_agent_settings().MOBILERUN_OPENAI_OAUTH_REFRESH_SKEW_SECONDS
            )
        kwargs.setdefault(
            "timeout",
            get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS,
        )
        kwargs.setdefault("api_base", get_agent_settings().MOBILERUN_OPENAI_OAUTH_API_BASE)
        # This adapter collects Responses text, without forwarding Chat
        # Completions response_format or function tools. Include the schema
        # in the prompt and validate the returned JSON through LlamaIndex.
        kwargs.setdefault("pydantic_program_mode", PydanticProgramMode.LLM)
        super().__init__(model=resolved_model, api_key=seed_api_key, **kwargs)
        self._oauth_refresh_skew_ms = max(0, int(oauth_refresh_skew_seconds)) * 1000
        self._use_chatgpt_account_header = use_chatgpt_account_header
        self._oauth_account_id = oauth_account_id
        self._active_access_token = None
        self._responses_api_base = get_agent_settings().MOBILERUN_OPENAI_OAUTH_CODEX_API_BASE

        store = OpenAIOAuthCredentialStore(
            oauth_credential_path or get_agent_settings().openai_oauth_credential_path
        )
        self._oauth_manager = OpenAIOAuthSessionManager(
            issuer=oauth_issuer,
            client_id=oauth_client_id,
            credential_store=store,
            request_timeout=self.timeout,
            http_client=self._http_client,
        )

        if oauth_access_token or oauth_refresh_token:
            self._oauth_manager.set_initial_credentials(
                OpenAIOAuthCredentials(
                    access_token=oauth_access_token or seed_api_key,
                    refresh_token=oauth_refresh_token,
                    expires_at_ms=oauth_expires_at_ms,
                    account_id=oauth_account_id,
                )
            )

    def _ensure_access_token(self) -> OpenAIOAuthCredentials:
        creds = self._oauth_manager.get_valid_credentials(skew_ms=self._oauth_refresh_skew_ms)

        if self._active_access_token != creds.access_token:
            self._active_access_token = creds.access_token
            self.api_key = creds.access_token
            self._client = None
            self._aclient = None

        if creds.account_id and not self._oauth_account_id:
            self._oauth_account_id = creds.account_id

        if self.api_base != self._responses_api_base:
            self.api_base = self._responses_api_base
            self._client = None
            self._aclient = None

        return creds

    def _get_client(self) -> OpenAIClient:
        self._ensure_access_token()
        return super()._get_client()

    def _get_aclient(self) -> AsyncOpenAI:
        self._ensure_access_token()
        return super()._get_aclient()

    def _get_credential_kwargs(self, is_async: bool = False) -> Dict[str, Any]:
        creds = self._ensure_access_token()
        kwargs = super()._get_credential_kwargs(is_async=is_async)
        kwargs["api_key"] = creds.access_token

        headers = dict(self.default_headers or {})
        headers.setdefault("originator", "codex_cli_rs")
        headers.setdefault("Accept", "application/json")
        account_id = self._oauth_account_id or creds.account_id
        if self._use_chatgpt_account_header and account_id:
            headers["ChatGPT-Account-Id"] = account_id
        kwargs["default_headers"] = headers or None

        return kwargs

    def _resolve_codex_instructions(self, messages: list[ChatMessage]) -> str:
        system_parts: list[str] = []
        for msg in messages:
            if msg.role != MessageRole.SYSTEM:
                continue
            content = msg.content
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
        return "\n\n".join(system_parts) if system_parts else "You are a helpful coding assistant."

    def _build_responses_payload(self, messages: list[ChatMessage]) -> ResponseInputParam:
        non_system_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        try:
            payload_raw = to_openai_message_dicts(
                non_system_messages,
                model=self.model,
                is_responses_api=True,
                store=False,
            )
        except TypeError:
            payload_raw = to_openai_message_dicts(
                non_system_messages,
                model=self.model,
                is_responses_api=True,
            )

        if isinstance(payload_raw, str):
            payload: list[dict[str, Any]] = [{"role": "user", "content": payload_raw}]
        else:
            # LlamaIndex's return annotation misses its is_responses_api=True branch.
            payload = cast(list[dict[str, Any]], payload_raw)

        normalized: list[dict[str, Any]] = []
        for item in payload:
            role = str(item.get("role", "user"))
            content = item.get("content")

            if isinstance(content, str):
                text_type = "input_text" if role == "user" else "output_text"
                normalized.append({**item, "content": [{"type": text_type, "text": content}]})
                continue

            if isinstance(content, list):
                fixed: list[Any] = []
                for entry in content:
                    if (
                        isinstance(entry, dict)
                        and entry.get("type") == "text"
                        and isinstance(entry.get("text"), str)
                    ):
                        text_type = "input_text" if role == "user" else "output_text"
                        fixed.append({**entry, "type": text_type})
                    else:
                        fixed.append(entry)
                normalized.append({**item, "content": fixed})
                continue

            normalized.append(item)

        # LlamaIndex incorrectly annotates its Responses API output as chat-completion input.
        return cast(ResponseInputParam, normalized)

    def _sanitize_reasoning_kwargs(self, runtime_kwargs: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            merged["reasoning"] = {"effort": self.reasoning_effort}
        merged.update(self.additional_kwargs or {})
        merged.update(runtime_kwargs)

        if self.model in OPENAI_GPT6_MODELS:
            if self.model == _GPT_6_ASTRA_MODEL:
                merged.setdefault("reasoning", {"effort": OPENAI_ASTRA_DEFAULT_REASONING_EFFORT})
            for key in _GPT_6_UNSUPPORTED_PARAMS:
                merged.pop(key, None)
            include = merged.get("include")
            if isinstance(include, (list, tuple)):
                filtered_include = [
                    value for value in include if value != _GPT_6_UNSUPPORTED_INCLUDE
                ]
                merged["include"] = filtered_include or None

        supported_efforts = OPENAI_REASONING_EFFORTS.get(self.model)
        if supported_efforts is None:
            return merged
        reasoning = merged.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, dict):
            raise ValueError(
                f"{self.model} reasoning must be a mapping with an optional 'effort' value."
            )
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        if effort is not None and effort not in supported_efforts:
            supported = ", ".join(sorted(supported_efforts))
            raise ValueError(
                f"{self.model} does not support reasoning effort "
                f"{effort!r}; use one of: {supported}."
            )
        return merged

    def _collect_stream_text_sync(
        self, events: Stream[ResponseStreamEvent]
    ) -> tuple[str, Response | None, list[ToolCallBlock]]:
        text_parts: list[str] = []
        final_response: Response | None = None
        tool_calls: list[ToolCallBlock] = []
        try:
            for event in events:
                tool_calls.extend(self._done_function_calls(event))
                if event.type == "response.output_text.delta":
                    delta = event.delta
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                elif event.type == "response.completed":
                    final_response = event.response
        finally:
            events.close()

        collected = "".join(text_parts)
        if collected:
            return collected, final_response, tool_calls

        if final_response is not None:
            final_text = final_response.output_text
            if isinstance(final_text, str) and final_text:
                return final_text, final_response, tool_calls

        return collected, final_response, tool_calls

    async def _collect_stream_text_async(
        self, events: AsyncStream[ResponseStreamEvent]
    ) -> tuple[str, Response | None, list[ToolCallBlock]]:
        text_parts: list[str] = []
        final_response: Response | None = None
        tool_calls: list[ToolCallBlock] = []
        try:
            async for event in events:
                tool_calls.extend(self._done_function_calls(event))
                if event.type == "response.output_text.delta":
                    delta = event.delta
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                elif event.type == "response.completed":
                    final_response = event.response
        finally:
            await events.close()

        collected = "".join(text_parts)
        if collected:
            return collected, final_response, tool_calls

        if final_response is not None:
            final_text = final_response.output_text
            if isinstance(final_text, str) and final_text:
                return final_text, final_response, tool_calls

        return collected, final_response, tool_calls

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        # Provider and transport exceptions expose different optional status fields.
        status_code = getattr(exc, "status_code", None)
        if status_code == 404:
            return True
        response = getattr(exc, "response", None)  # Third-party exceptions vary.
        if response is not None and getattr(response, "status_code", None) == 404:  # SDKs vary.
            return True
        return "404" in str(exc) and "not found" in str(exc).lower()

    def _codex_request_kwargs(
        self, messages: list[ChatMessage], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        kwargs = self._sanitize_reasoning_kwargs(kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": self._resolve_codex_instructions(messages),
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
            "stream": True,
        }
        for key in (
            "reasoning",
            "include",
            "service_tier",
            "text",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ):
            if key in kwargs and kwargs[key] is not None:
                request_kwargs[key] = kwargs[key]
        return request_kwargs

    @staticmethod
    def _function_call_blocks(
        items: Iterable[ResponseOutputItem] | None,
    ) -> list[ToolCallBlock]:
        return [
            ToolCallBlock(
                tool_name=item.name,
                tool_call_id=item.call_id,
                tool_kwargs=item.arguments,
            )
            for item in items or []
            if item.type == "function_call"
        ]

    @classmethod
    def _response_message(
        cls, text: str, response: Response | None, tool_calls: list[ToolCallBlock]
    ) -> ChatMessage:
        # Codex may omit output from the completed event, so calls are
        # collected from output_item.done events first.
        # Completed-event payloads can omit output on provider-compatible servers.
        tool_calls = tool_calls or cls._function_call_blocks(getattr(response, "output", None))
        if not tool_calls:
            return ChatMessage(role=MessageRole.ASSISTANT, content=text)
        blocks = [TextBlock(text=text)] if text else []
        return ChatMessage(role=MessageRole.ASSISTANT, blocks=[*blocks, *tool_calls])

    @classmethod
    def _done_function_calls(cls, event: ResponseStreamEvent) -> list[ToolCallBlock]:
        if event.type != "response.output_item.done":
            return []
        return cls._function_call_blocks([event.item])

    def _should_fall_back_to_backend_api(self, exc: Exception) -> bool:
        return (
            self._responses_api_base == get_agent_settings().MOBILERUN_OPENAI_OAUTH_CODEX_API_BASE
            and (self._is_not_found_error(exc))
        )

    def _use_backend_api_base(self) -> None:
        self._responses_api_base = get_agent_settings().MOBILERUN_OPENAI_OAUTH_BACKEND_API_BASE
        self.api_base = self._responses_api_base
        self._client = None
        self._aclient = None

    @staticmethod
    def _stream_chunk(text: str, delta: str, raw: Any = None) -> ChatResponse:
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            delta=delta,
            raw=raw,
            additional_kwargs={},
        )

    @classmethod
    def _stream_event_chunk(cls, event: ResponseStreamEvent, text: str) -> ChatResponse | None:
        if event.type == "response.output_text.delta":
            delta = event.delta
            if isinstance(delta, str) and delta:
                return cls._stream_chunk(text + delta, delta)
        elif event.type == "response.completed":
            response = event.response
            final_text = response.output_text
            if not text and isinstance(final_text, str) and final_text:
                return cls._stream_chunk(final_text, final_text, raw=response)
            return cls._stream_chunk(text, "", raw=response)
        return None

    @staticmethod
    def _structured_request_kwargs(
        output_cls: Any, llm_kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        kwargs = dict(llm_kwargs or {})
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": output_cls.__name__,
                "schema": to_strict_json_schema(output_cls),
                "strict": True,
            }
        }
        kwargs["_structured_output"] = True
        return kwargs

    def structured_predict(
        self,
        output_cls: Any,
        prompt: Any,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Any:
        response = self._chat(
            prompt.format_messages(**prompt_args),
            **self._structured_request_kwargs(output_cls, llm_kwargs),
        )
        return output_cls.model_validate_json(response.message.content or "")

    async def astructured_predict(
        self,
        output_cls: Any,
        prompt: Any,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Any:
        response = await self._achat(
            prompt.format_messages(**prompt_args),
            **self._structured_request_kwargs(output_cls, llm_kwargs),
        )
        return output_cls.model_validate_json(response.message.content or "")

    @llm_retry_decorator
    def _chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        structured = kwargs.pop("_structured_output", False)
        request_kwargs = self._codex_request_kwargs(messages, kwargs)
        if structured:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                request_kwargs.pop(key, None)
        self._ensure_access_token()
        client = self._get_client()
        payload = self._build_responses_payload(messages)

        try:
            events = client.responses.create(input=payload, **request_kwargs)
            text, response, tool_calls = self._collect_stream_text_sync(events)
        except Exception as exc:
            if self._should_fall_back_to_backend_api(exc):
                self._use_backend_api_base()
                client = self._get_client()
                events = client.responses.create(input=payload, **request_kwargs)
                text, response, tool_calls = self._collect_stream_text_sync(events)
            else:
                raise

        return ChatResponse(
            message=self._response_message(text, response, tool_calls),
            raw=response,
            additional_kwargs={},
        )

    @llm_retry_decorator
    async def _achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        structured = kwargs.pop("_structured_output", False)
        request_kwargs = self._codex_request_kwargs(messages, kwargs)
        if structured:
            for key in ("tools", "tool_choice", "parallel_tool_calls"):
                request_kwargs.pop(key, None)
        self._ensure_access_token()
        aclient = self._get_aclient()
        payload = self._build_responses_payload(messages)

        try:
            events = await aclient.responses.create(input=payload, **request_kwargs)
            text, response, tool_calls = await self._collect_stream_text_async(events)
        except Exception as exc:
            if self._should_fall_back_to_backend_api(exc):
                self._use_backend_api_base()
                aclient = self._get_aclient()
                events = await aclient.responses.create(input=payload, **request_kwargs)
                text, response, tool_calls = await self._collect_stream_text_async(events)
            else:
                raise

        return ChatResponse(
            message=self._response_message(text, response, tool_calls),
            raw=response,
            additional_kwargs={},
        )

    # The base class streams through Chat Completions, which Codex rejects.
    @llm_retry_decorator
    def _stream_chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        request_kwargs = self._codex_request_kwargs(messages, kwargs)
        self._ensure_access_token()
        payload = self._build_responses_payload(messages)
        try:
            events = self._get_client().responses.create(input=payload, **request_kwargs)
        except Exception as exc:
            if not self._should_fall_back_to_backend_api(exc):
                raise
            self._use_backend_api_base()
            events = self._get_client().responses.create(input=payload, **request_kwargs)

        def gen() -> ChatResponseGen:
            text = ""
            tool_calls: list[ToolCallBlock] = []
            try:
                for event in events:
                    tool_calls.extend(self._done_function_calls(event))
                    chunk = self._stream_event_chunk(event, text)
                    if chunk is not None:
                        text = chunk.message.content or ""
                        if event.type == "response.completed":
                            chunk.message = self._response_message(text, chunk.raw, tool_calls)
                        yield chunk
            finally:
                events.close()

        return gen()

    @llm_retry_decorator
    async def _astream_chat(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        request_kwargs = self._codex_request_kwargs(messages, kwargs)
        self._ensure_access_token()
        payload = self._build_responses_payload(messages)
        try:
            events = await self._get_aclient().responses.create(input=payload, **request_kwargs)
        except Exception as exc:
            if not self._should_fall_back_to_backend_api(exc):
                raise
            self._use_backend_api_base()
            events = await self._get_aclient().responses.create(input=payload, **request_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            text = ""
            tool_calls: list[ToolCallBlock] = []
            try:
                async for event in events:
                    tool_calls.extend(self._done_function_calls(event))
                    chunk = self._stream_event_chunk(event, text)
                    if chunk is not None:
                        text = chunk.message.content or ""
                        if event.type == "response.completed":
                            chunk.message = self._response_message(text, chunk.raw, tool_calls)
                        yield chunk
            finally:
                await events.close()

        return gen()

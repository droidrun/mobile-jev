"""xAI subscription OAuth transport for the Responses API.

Credentials are owned by Mobilerun and stored in its shared auth profile.  The
OAuth client and inference proxy values are deliberately pinned: accepting
caller-controlled endpoints would let an attacker exfiltrate refresh tokens.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import httpx
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.llms.openai.responses import OpenAIResponses
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai.types.responses.response_input_param import ResponseInputParam

from mobilerun.agents.clients.oauth.grok import (
    _DEVICE_GRANT as _DEVICE_GRANT,
)
from mobilerun.agents.clients.oauth.grok import (
    _DEVICE_SURFACE as _DEVICE_SURFACE,
)
from mobilerun.agents.clients.oauth.grok import (
    _DEVICE_SURFACE_HEADER as _DEVICE_SURFACE_HEADER,
)
from mobilerun.agents.clients.oauth.grok import (
    _OAUTH_ERROR_CODE as _OAUTH_ERROR_CODE,
)
from mobilerun.agents.clients.oauth.grok import (
    _TOKEN_ERROR_CODES as _TOKEN_ERROR_CODES,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_CONTEXT_WINDOW as DEFAULT_GROK_CONTEXT_WINDOW,
)

# The proxy rejects /responses from clients below its minimum Grok Build
# version (HTTP 426). Keep it pinned rather than deriving it from any installed CLI.
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_MODEL as DEFAULT_GROK_MODEL,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_AUTHORIZE_URL as DEFAULT_GROK_OAUTH_AUTHORIZE_URL,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_CALLBACK_HOST as DEFAULT_GROK_OAUTH_CALLBACK_HOST,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_CALLBACK_PATH as DEFAULT_GROK_OAUTH_CALLBACK_PATH,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_CLIENT_ID as DEFAULT_GROK_OAUTH_CLIENT_ID,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_DEVICE_URL as DEFAULT_GROK_OAUTH_DEVICE_URL,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_ISSUER as DEFAULT_GROK_OAUTH_ISSUER,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_JWKS_URL as DEFAULT_GROK_OAUTH_JWKS_URL,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_PROXY as DEFAULT_GROK_OAUTH_PROXY,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_SCOPES as DEFAULT_GROK_OAUTH_SCOPES,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_SLOT as DEFAULT_GROK_OAUTH_SLOT,
)
from mobilerun.agents.clients.oauth.grok import (
    DEFAULT_GROK_OAUTH_TOKEN_URL as DEFAULT_GROK_OAUTH_TOKEN_URL,
)
from mobilerun.agents.clients.oauth.grok import (
    GROK_CLI_COMPAT_VERSION as GROK_CLI_COMPAT_VERSION,
)
from mobilerun.agents.clients.oauth.grok import (
    GROK_CLI_COMPAT_VERSION_HEADER as GROK_CLI_COMPAT_VERSION_HEADER,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokIDTokenValidator as GrokIDTokenValidator,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthAuth as GrokOAuthAuth,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthCredentials as GrokOAuthCredentials,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthCredentialStore as GrokOAuthCredentialStore,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthError as GrokOAuthError,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthReloginRequired as GrokOAuthReloginRequired,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthSession as GrokOAuthSession,
)
from mobilerun.agents.clients.oauth.grok import (
    GrokOAuthSessionManager as GrokOAuthSessionManager,
)
from mobilerun.agents.clients.oauth.grok import (
    _b64_no_pad as _b64_no_pad,
)
from mobilerun.agents.clients.oauth.grok import (
    _is_headless_environment as _is_headless_environment,
)
from mobilerun.agents.clients.oauth.grok import (
    _parse_callback_query as _parse_callback_query,
)
from mobilerun.agents.clients.oauth.grok import (
    _pkce_pair as _pkce_pair,
)
from mobilerun.agents.clients.oauth.grok import (
    _safe_error_code as _safe_error_code,
)
from mobilerun.agents.clients.oauth.grok import (
    _safe_json_object as _safe_json_object,
)
from mobilerun.agents.clients.oauth.grok import (
    _safe_token_error_code as _safe_token_error_code,
)
from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.llm.providers.grok import (
    GROK_OAUTH_LEGACY_MODELS,
    GROK_OAUTH_MODELS,
    normalize_grok_model_id,
    sanitize_grok_responses_kwargs,
)


class GrokOAuth(OpenAIResponses, GrokOAuthSession):
    """LlamaIndex Responses adapter backed only by Mobilerun's xAI OAuth slot."""

    supports_native_tool_calls: ClassVar[bool] = True

    @classmethod
    def class_name(cls) -> str:
        return "GrokOAuth"

    def __init__(
        self,
        model: str = DEFAULT_GROK_MODEL,
        oauth_credential_path: str | None = None,
        credential_path: str | None = None,
        oauth_access_token: str | None = None,
        oauth_refresh_token: str | None = None,
        oauth_expires_at_ms: int | None = None,
        oauth_refresh_skew_seconds: int | None = None,
        oauth_session_manager: GrokOAuthSessionManager | None = None,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        model = normalize_grok_model_id(model)
        if model not in (*GROK_OAUTH_MODELS, *GROK_OAUTH_LEGACY_MODELS):
            raise ValueError(
                f"Model {model!r} is not supported with XAI OAuth. "
                f"Use {', '.join(GROK_OAUTH_MODELS)}."
            )
        path = (
            oauth_credential_path
            or credential_path
            or str(get_agent_settings().grok_oauth_credential_path)
        )
        manager = oauth_session_manager or GrokOAuthSessionManager(
            credential_store=GrokOAuthCredentialStore(path),
            request_timeout=float(
                kwargs.get(
                    "timeout",
                    get_agent_settings().MOBILERUN_GROK_OAUTH_DEFAULT_TIMEOUT_SECONDS,
                )
            ),
            refresh_skew_seconds=oauth_refresh_skew_seconds,
        )
        if oauth_access_token or oauth_refresh_token:
            manager.set_initial_credentials(
                GrokOAuthCredentials(
                    access_token=oauth_access_token or "oauth",
                    refresh_token=oauth_refresh_token,
                    expires_at_ms=(oauth_expires_at_ms if oauth_access_token else 0),
                )
            )

        auth = GrokOAuthAuth(manager, model=model)
        if http_client is None:
            http_client = httpx.Client(auth=auth)
        else:
            http_client.auth = auth
        if async_http_client is None:
            async_http_client = httpx.AsyncClient(auth=auth)
        else:
            async_http_client.auth = auth

        supplied_headers = dict(kwargs.pop("default_headers", None) or {})
        supplied_headers.update(
            {
                "X-XAI-Token-Auth": "xai-grok-cli",
                "x-grok-model-override": model,
                GROK_CLI_COMPAT_VERSION_HEADER: GROK_CLI_COMPAT_VERSION,
            }
        )
        kwargs.pop("api_key", None)
        kwargs.pop("api_base", None)
        kwargs.pop("base_url", None)
        kwargs.pop("store", None)
        kwargs.pop("track_previous_responses", None)
        kwargs.pop("reasoning_options", None)
        kwargs.pop("context_window", None)
        kwargs.pop("openai_client", None)
        kwargs.pop("async_openai_client", None)
        super().__init__(
            model=model,
            api_key="oauth",
            api_base=get_agent_settings().MOBILERUN_GROK_OAUTH_PROXY,
            context_window=DEFAULT_GROK_CONTEXT_WINDOW,
            store=False,
            track_previous_responses=False,
            reasoning_options=None,
            default_headers=supplied_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            **kwargs,
        )
        self._oauth_manager = manager

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=DEFAULT_GROK_CONTEXT_WINDOW,
            num_output=self.max_output_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )

    @property
    def _tokenizer(self) -> None:
        # tiktoken does not have an encoding registered for xAI model ids.
        return None

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        return self._sanitize_request_kwargs(model_kwargs)

    def _sanitize_request_kwargs(
        self,
        model_kwargs: dict[str, Any],
        *,
        omit_tool_choice: bool = False,
    ) -> dict[str, Any]:
        sanitized = dict(
            sanitize_grok_responses_kwargs(
                dict(model_kwargs),
                omit_sampler_fields=True,
                omit_tool_choice=omit_tool_choice,
            )
        )
        # Both normal kwargs and ``extra_body`` are merged after constructor
        # defaults. Pin the payload model as well as the proxy override header.
        sanitized["model"] = self.model
        return sanitized

    def structured_predict(
        self,
        output_cls: Any,
        prompt: Any,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Any:
        sanitized = self._sanitize_request_kwargs(dict(llm_kwargs or {}), omit_tool_choice=True)
        sanitized.pop("model", None)
        sanitized.pop("tool_choice", None)
        messages = prompt.format_messages(**prompt_args)
        message_dicts = to_openai_message_dicts(messages, model=self.model, is_responses_api=True)
        response = self._client.responses.parse(
            model=self._responses_model,
            # LlamaIndex's annotation omits the Responses API return type.
            input=cast(str | ResponseInputParam, message_dicts),
            text_format=output_cls,
            **sanitized,
        )
        if response.output_parsed is not None:
            return response.output_parsed
        raise ValueError("Failed to produce a structured response from the model.")

    async def astructured_predict(
        self,
        output_cls: Any,
        prompt: Any,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Any:
        sanitized = self._sanitize_request_kwargs(dict(llm_kwargs or {}), omit_tool_choice=True)
        sanitized.pop("model", None)
        sanitized.pop("tool_choice", None)
        messages = prompt.format_messages(**prompt_args)
        message_dicts = to_openai_message_dicts(messages, model=self.model, is_responses_api=True)
        response = await self._aclient.responses.parse(
            model=self._responses_model,
            # LlamaIndex's annotation omits the Responses API return type.
            input=cast(str | ResponseInputParam, message_dicts),
            text_format=output_cls,
            **sanitized,
        )
        if response.output_parsed is not None:
            return response.output_parsed
        raise ValueError("Failed to produce a structured response from the model.")


# Descriptive alias for callers that prefer the full class name.

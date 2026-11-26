"""Bound LlamaIndex chat calls; only invalid replies are re-asked here."""

import asyncio
import logging
import traceback
from collections.abc import Callable, Sequence
from contextlib import aclosing, nullcontext
from typing import Any

import anthropic
import httpx
import openai
from google.genai.errors import APIError as GeminiAPIError
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, TextBlock
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool
from mobilerun.errors.base import MobilerunError
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from openinference.instrumentation import suppress_tracing
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig

_DEFAULT_TIMEOUT = DecisionConfig().timeout


class InvalidResponse(ValueError):
    """A model-correctable problem. Its message must be safe to send to the model."""

    usage: "Usage | None" = None


class _RequestError(MobilerunError, RuntimeError):
    """Safe for the base CLI renderer; details are debug-only exception notes."""


class _BadRequestBody(BaseModel):
    param: str | None = None
    code: str | None = None
    type: str | None = None


_REQUEST_FIELDS = (
    ResponseCreateParamsBase.__annotations__.keys()
    | CompletionCreateParamsBase.__annotations__.keys()
)
_REQUEST_BODY = TypeAdapter(dict[str, object])
_BAD_REQUEST_TAGS = frozenset(
    {
        "invalid_request_error",
        "invalid_value",
        "invalid_type",
        "invalid_parameter",
        "unsupported_parameter",
        "unsupported_value",
        "missing_required_parameter",
        "invalid_json_schema",
        "model_not_found",
        "context_length_exceeded",
    }
)


def _bad_request(exc: openai.BadRequestError, log_response: bool) -> _RequestError:
    try:
        body = _BadRequestBody.model_validate(exc.body)
    except ValidationError:
        return _RequestError("LLM request failed (400)")
    field = ""
    if body.param in _REQUEST_FIELDS:
        try:
            sent = _REQUEST_BODY.validate_json(exc.request.content)
        except (ValidationError, RuntimeError):  # Body unavailable or not a JSON object.
            pass
        else:
            if body.param in sent:
                field = f", param: {body.param}"
    error = _RequestError(f"LLM request failed (400{field})")
    if log_response:
        for label, value in (("code", body.code), ("type", body.type)):
            if value in _BAD_REQUEST_TAGS:
                error.add_note(f"{label}: {value}")
    return error


class Usage(BaseModel):
    """Counts per received reply. None means unreported, not zero."""

    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    requests: int = 1


class _WireModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class _Details(_WireModel):
    reasoning_tokens: int | None = None


class _Counts(_WireModel):
    inputs: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "promptTokenCount",
            "prompt_eval_count",
        ),
    )
    outputs: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
            "eval_count",
        ),
    )
    total: int | None = Field(
        default=None,
        validation_alias=AliasChoices("total_tokens", "total_token_count", "totalTokenCount"),
    )
    thoughts: int | None = Field(
        default=None, validation_alias=AliasChoices("thoughts_token_count", "thoughtsTokenCount")
    )
    details: _Details | None = Field(
        default=None,
        validation_alias=AliasChoices("output_tokens_details", "completion_tokens_details"),
    )

    def normalized(self) -> Usage:
        outputs = self.outputs
        if self.thoughts is not None:
            outputs = (outputs or 0) + self.thoughts
        elif self.details and self.inputs is not None and outputs is not None:
            # Some providers exclude reasoning from output, but include it in total.
            reasoning = self.details.reasoning_tokens or 0
            if self.total == self.inputs + outputs + reasoning:
                outputs += reasoning
        total = self.total
        if total is None and self.inputs is not None and outputs is not None:
            total = self.inputs + outputs
        return Usage(
            request_tokens=self.inputs,
            response_tokens=outputs,
            total_tokens=total,
            reasoning_tokens=(
                self.thoughts
                if self.thoughts is not None
                else self.details.reasoning_tokens if self.details else None
            ),
        )


class _Payload(_WireModel):
    usage: object | None = Field(
        default=None, validation_alias=AliasChoices("usage_metadata", "usageMetadata", "usage")
    )


class _Envelope(_Payload):
    response: _Payload | None = None


def usage_from_response(reply: ChatResponse) -> Usage:
    """Read wire usage before adapter-normalized fallback (which can omit thinking)."""
    for source in (reply.raw, reply.additional_kwargs):
        try:
            envelope = _Envelope.model_validate(source)
        except ValidationError:
            continue
        for candidate in (envelope.usage, envelope.response and envelope.response.usage, source):
            if candidate is None:
                continue
            try:
                counts = _Counts.model_validate(candidate)
            except ValidationError:
                continue
            if counts.model_fields_set:
                return counts.normalized()
    return Usage()


async def _stream(
    llm: LLM,
    history: Sequence[ChatMessage],
    options: dict[str, Any],
    tools: Sequence[BaseTool] | None = None,
) -> ChatResponse:
    last: ChatResponse | None = None
    message = ChatMessage(role="assistant")
    deltas: list[str] = []
    match llm:
        case FunctionCallingLLM() if tools is not None:
            stream = await llm.astream_chat_with_tools(tools, chat_history=list(history), **options)
        case _:
            stream = await llm.astream_chat(
                history, **({"tools": tools} if tools is not None else {}), **options
            )
    async with aclosing(stream) as chunks:
        async for last in chunks:
            if last.delta:
                deltas.append(last.delta)
            if (
                last.message.content
                or last.message.additional_kwargs.get("tool_calls")
                or any(b.block_type != "text" for b in last.message.blocks)
            ):
                message = last.message

    # Some adapters send only the current delta as message content; others send
    # cumulative blocks. Keep cumulative blocks intact, including tool IDs.
    if deltas and (not message.content or (len(deltas) > 1 and message.content == deltas[-1])):
        text = TextBlock(text="".join(deltas))
        blocks = [text if b.block_type == "text" else b for b in message.blocks]
        if not any(b.block_type == "text" for b in blocks):
            blocks.append(text)
        message = message.model_copy(update={"blocks": blocks})
    return last.model_copy(update={"message": message}) if last else ChatResponse(message=message)


async def achat(
    llm: LLM,
    messages: Sequence[ChatMessage],
    timeout: float = _DEFAULT_TIMEOUT,
    stream: bool = False,
    tools: Sequence[BaseTool] | None = None,
    log_response: bool = True,
    on_usage: Callable[[Usage], None] | None = None,
    **options: Any,
) -> ChatResponse:
    """One call, including stream consumption. Never expose provider error bodies."""
    try:
        async with nullcontext() if log_response else suppress_tracing(), asyncio.timeout(timeout):
            if stream:
                reply = await _stream(llm, messages, options, tools)
            else:
                match llm:
                    case FunctionCallingLLM() if tools is not None:
                        if options.get("allow_parallel_tool_calls") is False:
                            # Public achat_with_tools discards additional returned calls in single-call mode.
                            # Let the adapter prepare the request, then inspect the untouched reply.
                            prepared = llm._prepare_chat_with_tools_compat(
                                tools, chat_history=list(messages), **options
                            )
                            reply = await llm.achat(**prepared)
                        else:
                            reply = await llm.achat_with_tools(
                                tools, chat_history=list(messages), **options
                            )
                    case _:
                        reply = await llm.achat(
                            messages, **({"tools": tools} if tools is not None else {}), **options
                        )
    except TimeoutError:
        raise TimeoutError("LLM request timed out") from None
    except (
        ConnectionError,
        httpx.TransportError,
        openai.APIConnectionError,
        anthropic.APIConnectionError,
    ):
        raise ConnectionError("LLM connection failed") from None
    except openai.BadRequestError as bad_request:
        raise _bad_request(bad_request, log_response=log_response) from None
    except (openai.APIStatusError, anthropic.APIStatusError) as status:
        if status.status_code == 429 or status.status_code >= 500:
            raise ConnectionError("LLM temporarily unavailable") from None
        raise RuntimeError(f"LLM request failed ({status.status_code})") from None
    except GeminiAPIError as gemini:
        if gemini.code == 429 or gemini.code >= 500:
            raise ConnectionError("LLM temporarily unavailable") from None
        raise RuntimeError(f"LLM request failed ({gemini.code})") from None
    except Exception as exc:
        # SDK messages may include unfiltered server bodies. Keep stack locations,
        # not source lines, request data, exception text or frame locals.
        frames = " -> ".join(
            f"{frame.f_code.co_filename}:{line} in {frame.f_code.co_name}"
            for frame, line in traceback.walk_tb(exc.__traceback__)
        )
        error = RuntimeError(f"LLM request failed ({type(exc).__name__})")
        error.add_note(frames)
        logging.getLogger("mobilerun").debug("%s: %s", error, frames)
        raise error from None
    if on_usage is not None:
        on_usage(usage_from_response(reply))
    if not (reply.message.content or "").strip() and not (
        tools is not None
        and (
            any(b.block_type == "tool_call" for b in reply.message.blocks)
            or reply.message.additional_kwargs.get("tool_calls")
        )
    ):
        error_reply = InvalidResponse(
            render_prompt("assets/prompts/providers/empty_response.jinja2")
        )
        error_reply.usage = usage_from_response(reply)
        raise error_reply
    return reply

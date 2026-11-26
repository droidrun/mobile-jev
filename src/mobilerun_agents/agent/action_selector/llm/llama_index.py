"""LlamaIndex transport and usage accounting for the shared LLM pipeline."""

from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM

from mobilerun_agents.agent.action_selector.base import (
    ActionSelection,
    NativeToolAPI,
    Vision,
)
from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.config.user_config import DecisionConfig

from .calls import achat, usage_from_response
from .factory import adapter_capabilities
from .models import Capabilities
from .parsing import ToolCall, ToolCallParseResult, ToolCallParseStatus


class LlamaIndexLLMSelector(LLMSelector):
    @property
    def diagnostic_name(self) -> str:
        metadata = self.llm.metadata
        adapter = type(self.llm)
        module = adapter.__module__.split(".")
        provider = (
            module[2]
            if module[:2] == ["llama_index", "llms"] and len(module) > 2
            else adapter.__name__
        )
        return f"llm:{provider}/{metadata.model_name}"

    def __init__(
        self,
        llm: LLM,
        vision: Vision = "on",
        supports_native: bool | None = None,
        native_tool_api: NativeToolAPI | None = None,
        image_limits: Capabilities | None = None,
        config: DecisionConfig | None = None,
    ) -> None:
        super().__init__(vision, config)
        self.llm = llm
        self._image_limits = image_limits or adapter_capabilities(llm)
        declared = native_tool_api
        if supports_native and declared is None:
            from llama_index.llms.anthropic import Anthropic
            from llama_index.llms.google_genai import GoogleGenAI

            match llm:
                case Anthropic():
                    declared = "anthropic"
                case GoogleGenAI():
                    declared = "gemini"
        self._supports_native_tool_calls = (
            declared is not None if supports_native is None else supports_native
        )
        self._native_tool_api = native_tool_api or declared or "responses"

    @property
    def supports_native_tool_calls(self) -> bool:
        return self._supports_native_tool_calls

    @property
    def native_tool_api(self) -> NativeToolAPI:
        return self._native_tool_api

    @property
    def shared_llm(self) -> LLM:
        return self.llm

    async def _achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse | None:
        return await achat(self.llm, messages, **kwargs)

    def _native_calls(self, response: ChatResponse) -> ToolCallParseResult:
        if self.native_tool_api in {"responses", "chat_completions"}:
            return super()._native_calls(response)
        match self.llm:
            case FunctionCallingLLM() as llm:
                try:
                    calls = llm.get_tool_calls_from_response(response)
                except (KeyError, TypeError, ValueError):
                    return ToolCallParseResult("", [], ToolCallParseStatus.MALFORMED)
                return ToolCallParseResult(
                    response.message.content or "",
                    [ToolCall(c.tool_name, c.tool_kwargs, call_id=c.tool_id) for c in calls],
                    ToolCallParseStatus.VALID,
                )
            case _:
                raise TypeError("Native extraction requires a function-calling adapter")

    def _to_selection(self, response: ChatResponse, parsed: ActionSelection) -> ActionSelection:
        selection = super()._to_selection(response, parsed)
        selection.usage = usage_from_response(response)
        return selection

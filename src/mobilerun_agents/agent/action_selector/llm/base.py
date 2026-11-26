"""Reusable chat selection pipeline with one required transport hook."""

import asyncio
import json
from abc import abstractmethod
from collections.abc import Mapping
from time import monotonic
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock
from llama_index.core.tools import BaseTool

from mobilerun_agents.agent.action_selector.base import (
    ActionSelection,
    ActionSelector,
    NativeToolAPI,
    Vision,
)
from mobilerun_agents.agent.action_selector.llm.parsing import (
    ToolCallParseResult,
    ToolCallParseStatus,
    coerce_arguments,
    extract_add_memory,
    parse_native_tool_calls,
    parse_tool_calls_detailed,
)
from mobilerun_agents.agent.logs.diagnostics import ModelCall, sink
from mobilerun_agents.agent.memory.messages import text_messages
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.base import RegistryTool
from mobilerun_agents.agent.tools.helpers import visual_tool_names
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig

from .calls import InvalidResponse, Usage, usage_from_response
from .image import ModelImage, prepare_image
from .models import Capabilities


class LLMSelector(ActionSelector):
    @property
    def supports_native_tool_calls(self) -> bool:
        return False

    @property
    def native_tool_api(self) -> NativeToolAPI:
        return "responses"

    abstention_reason = "LLM response is None. This is a critical error."

    def __init__(self, vision: Vision = "on", config: DecisionConfig | None = None) -> None:
        self.vision = vision
        self._image_limits = Capabilities()
        self.config = config or DecisionConfig()

    async def select(self, run: Run) -> ActionSelection:
        has_run_tools = bool(run.registry.by_name or run.selection_include)
        candidates = run.registry.tools(include=run.selection_include)
        coordinates, indexed = visual_tool_names(candidates)
        exclude = set(run.selection_exclude)
        if self.vision == "off":
            exclude |= coordinates
        if self.vision == "only" or not run.ui or not run.ui.elements:
            exclude |= indexed
        tools = run.registry.tools(exclude=exclude, include=run.selection_include)
        by_name = {member.metadata.get_name(): member for member in tools}
        data = None if self.vision == "off" else await run.screenshot()
        image = await asyncio.to_thread(prepare_image, run, self._image_limits, data)
        messages = await self._build_messages(run, image, by_name)
        kwargs: dict[str, Any] = {"stream": run.config.streaming, "timeout": self.config.timeout}
        for tool in tools:
            match tool:
                case RegistryTool(action=action) if action and action.sensitive_parameters:
                    kwargs["log_response"] = False
                    break
        if self.supports_native_tool_calls:
            kwargs.update(
                tools=tools,
                allow_parallel_tool_calls=run.config.parallel_tools,
                tool_required=self.native_tool_api == "responses",
            )
        diagnostic = sink.get()
        usage = Usage()
        started = monotonic()
        try:
            response = await self._achat(messages, **kwargs)
            if diagnostic and response is not None:
                usage = usage_from_response(response)
        except InvalidResponse as exc:
            usage = exc.usage or usage
            raise
        finally:
            if diagnostic:
                diagnostic(
                    ModelCall(
                        name=self.diagnostic_name,
                        image=(image.width, image.height) if image.data else None,
                        seconds=monotonic() - started,
                        inputs=usage.request_tokens,
                        outputs=usage.response_tokens,
                        reasoning=usage.reasoning_tokens,
                        total=usage.total_tokens,
                    )
                )
        if response is None:
            return ActionSelection(abstained=True)
        parsed = self._parse(response)
        if parsed.parse_result and parsed.parse_result.status == ToolCallParseStatus.MALFORMED:
            parsed.parse_error = render_prompt(self.config.invalid_arguments_prompt)
        if has_run_tools and not parsed.parse_error:
            for call in parsed.actions:
                member = by_name.get(call.name)
                try:
                    if member is None or member.metadata.fn_schema is None or call.error:
                        raise ValueError
                    call.parameters = coerce_arguments(
                        member.metadata.get_parameters_dict(),
                        call.parameters,
                        textual=not self.supports_native_tool_calls,
                    )
                    if call.parameters.keys() - member.metadata.fn_schema.model_fields.keys():
                        raise ValueError
                    member.metadata.fn_schema.model_validate_json(
                        json.dumps(call.parameters), strict=True
                    )
                    image.to_device(call, member.metadata.fn_schema.model_fields.keys())
                except (KeyError, ValueError, TypeError):
                    parsed.parse_error = render_prompt(
                        self.config.invalid_arguments_prompt
                    )
                    break
        # One repair owner: the workflow's Invalid -> Select edge.
        return self._to_selection(response, parsed)

    async def _build_messages(
        self, run: Run, image: ModelImage, names: Mapping[str, BaseTool]
    ) -> list[ChatMessage]:
        system = render_prompt(
            run.config.system_prompt,
            platform=run.device.platform,
            native_tools=self.supports_native_tool_calls,
            parallel_tools=run.config.parallel_tools,
            screenshot_only=self.vision == "only" or run.state_provider.requires_coordinate_tools,
            vision=image.data is not None,
            tool_descriptions="\n".join(
                f"{n}: {a.metadata.description} {a.metadata.fn_schema_str}"
                for n, a in names.items()
            ),
            available_tools=set(names),
            output_schema=None,
        )
        screen = "" if self.vision == "only" else run.ui.formatted_text
        if image.data:
            screen = render_prompt(
                "assets/prompts/llm/model_screen.jinja2",
                state=screen,
                screenshot_only=self.vision == "only"
                or run.state_provider.requires_coordinate_tools,
                width=image.width,
                height=image.height,
                normalized=image.normalized,
            )
        text = render_prompt(
            self.config.llm_request_prompt,
            state=screen,
        )
        user = ChatMessage(role="user", content=text or "Current screen unavailable.")
        if image.data:
            user.blocks.append(ImageBlock(image=image.data))
        memory = await run.history()
        if not self.supports_native_tool_calls or self.native_tool_api == "gemini":
            # Synthetic tool calls do not have Gemini's provider-issued thought signature.
            memory = text_messages(memory)
        system_message = ChatMessage(role="system", content=system)
        if memory and memory[0].role == "system":
            system_message.blocks.extend(memory.pop(0).blocks)
        return [system_message, *memory, user]

    @abstractmethod
    async def _achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse | None: ...

    def _parse(self, response: ChatResponse) -> ActionSelection:
        text = response.message.content or ""
        parsed = (
            self._native_calls(response)
            if self.supports_native_tool_calls
            else parse_tool_calls_detailed(text)
        )
        return ActionSelection(
            actions=parsed.calls,
            thought=parsed.thought,
            memory_update=extract_add_memory(parsed.thought),
            parse_result=parsed,
            textual_arguments=not self.supports_native_tool_calls,
        )

    def _native_calls(self, response: ChatResponse) -> ToolCallParseResult:
        return parse_native_tool_calls(response.message)

    def _to_selection(self, response: ChatResponse, parsed: ActionSelection) -> ActionSelection:
        parsed.response = response
        parsed.response_text = response.message.content or ""
        return parsed

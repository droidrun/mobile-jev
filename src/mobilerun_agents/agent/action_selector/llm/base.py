"""Build the chat request and validate the reply before exposing native calls."""

import asyncio
import json
from abc import abstractmethod
from typing import Any
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.devices import Device
from workflows import Context

from mobilerun_agents.agent.action_selector.base import (
    ActionSelector,
    NativeToolAPI,
    Selection,
    previous_action_feedback,
    screenshot_bytes,
)
from mobilerun_agents.agent.logs.memory import report_memory
from mobilerun_agents.agent.memory import read_memory
from mobilerun_agents.agent.tools.helpers import visual_tool_names
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig, Vision

from .calls import InvalidResponse
from .image import prepare_image
from .models import Capabilities
from .parsing import (
    ToolCallParseResult,
    ToolCallParseStatus,
    coerce_arguments,
    parse_native_tool_calls,
    parse_tool_calls_detailed,
)


class LLMSelector(ActionSelector):
    @property
    def supports_native_tool_calls(self) -> bool:
        return False

    @property
    def native_tool_api(self) -> NativeToolAPI:
        return "responses"

    def __init__(self, vision: Vision = "on", config: AgentConfig | None = None) -> None:
        self.vision = vision
        self._image_limits = Capabilities()
        self._config = config

    @property
    def config(self) -> AgentConfig:
        return self._config if self._config is not None else ConfigLoader.load().agent

    @config.setter
    def config(self, config: AgentConfig) -> None:
        self._config = config

    async def select(
        self, ctx: Context[AgentState], device: Device, memory: Memory, registry: ToolRegistry
    ) -> Selection:
        state = await ctx.store.get_state()
        snapshot = state.current
        if snapshot is None:
            raise ValueError("Selection requires a current observation")
        ui = snapshot.ui
        coordinates, indexed = visual_tool_names(registry.tools())
        exclude: set[str] = set()
        if self.vision == "off":
            exclude |= coordinates
        if self.vision == "only" or not ui.elements:
            exclude |= indexed
        tools = registry.tools(exclude)
        by_name = {member.metadata.get_name(): member for member in tools}
        data = None if self.vision == "off" else await screenshot_bytes(ctx, snapshot, device)
        coordinate_provider = device.state._resolved_provider.requires_coordinate_tools
        screenshot_only = self.vision == "only" or coordinate_provider
        image = await asyncio.to_thread(
            prepare_image, ui, coordinate_provider, self._image_limits, data
        )
        config = self.config
        decision = config.decision
        system = render_prompt(
            config.system_prompt,
            platform=device.platform,
            native_tools=self.supports_native_tool_calls,
            parallel_tools=config.parallel_tools,
            screenshot_only=screenshot_only,
            vision=image.data is not None,
            tool_descriptions="\n".join(
                f"{n}: {a.metadata.description} {a.metadata.fn_schema_str}"
                for n, a in by_name.items()
            ),
            available_tools=set(by_name),
        )
        screen = "" if self.vision == "only" else ui.formatted_text
        if image.data:
            screen = render_prompt(
                "assets/prompts/llm/model_screen.jinja2",
                state=screen,
                screenshot_only=screenshot_only,
                width=image.width,
                height=image.height,
                normalized=image.normalized,
            )
        text = render_prompt(
            decision.llm_request_prompt,
            state=screen,
            device_context=snapshot.model_dump(mode="json", include={"device_context"})[
                "device_context"
            ],
            repair=state.repair,
        )
        text = text or "Current screen unavailable."
        if previous := previous_action_feedback(state):
            text = f"{previous}\n{text}"
        user = ChatMessage(role="user", content=text)
        if image.data:
            user.blocks.append(ImageBlock(image=image.data))
        context = await read_memory(memory, ctx)
        if self.vision == "off":
            context = [
                ChatMessage(role=message.role, content=message.content) for message in context
            ]
        report_memory(ctx, memory, context, selector="llm")
        system_message = ChatMessage(role="system", content=system)
        if context and context[0].role == "system":
            # Native adapters expect a single textual system message. Prior-screen
            # images must stay in user content, before the current observation.
            images = []
            for block in context[0].blocks:
                match block:
                    case TextBlock():
                        system_message.blocks.append(block)
                    case ImageBlock():
                        images.append(block)
            context = context[1:]
            if images:
                context.insert(0, ChatMessage(role="user", blocks=images))
        kwargs: dict[str, Any] = {
            "stream": config.streaming,
            "timeout": decision.timeout,
            "log_response": not registry.has_sensitive,
        }
        if self.supports_native_tool_calls:
            kwargs.update(
                tools=tools,
                allow_parallel_tool_calls=config.parallel_tools,
                tool_required=self.native_tool_api == "responses",
            )
        response = await self._achat([system_message, *context, user], **kwargs)
        if response is None:
            return Selection()
        parsed = self._parse(response)
        error = render_prompt(decision.invalid_arguments_prompt)
        if parsed.status is ToolCallParseStatus.MALFORMED or not parsed.calls:
            raise InvalidResponse(error)
        calls: list[ToolSelection] = []
        for call in parsed.calls:
            member = by_name.get(call.name)
            try:
                if member is None or member.metadata.fn_schema is None or call.error:
                    raise ValueError
                schema = member.metadata.fn_schema
                argument_schema = member.metadata.get_parameters_dict()
                typed_output = call.name == "complete" and "output" in schema.model_fields
                if typed_output:
                    # Decode the container here; the workflow owns validation of its contents.
                    argument_schema["properties"] = {
                        **argument_schema["properties"],
                        "output": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                    }
                call.parameters = coerce_arguments(
                    argument_schema,
                    call.parameters,
                    textual=not self.supports_native_tool_calls,
                )
                if call.parameters.keys() - schema.model_fields.keys():
                    raise ValueError
                dispatch_arguments = (
                    {name: value for name, value in call.parameters.items() if name != "output"}
                    if typed_output
                    else call.parameters
                )
                schema.model_validate_json(json.dumps(dispatch_arguments), strict=True)
                image.to_device(call, schema.model_fields.keys())
            except (KeyError, ValueError, TypeError):
                raise InvalidResponse(error) from None
            calls.append(
                ToolSelection(
                    tool_id=call.call_id or f"call_{uuid4().hex}",
                    tool_name=call.name,
                    tool_kwargs=call.parameters,
                )
            )
        if not config.parallel_tools and len(calls) != 1:
            raise InvalidResponse(error)
        return Selection(calls=calls, thought=parsed.thought or None)

    @abstractmethod
    async def _achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse | None: ...

    def _parse(self, response: ChatResponse) -> ToolCallParseResult:
        if self.supports_native_tool_calls:
            return self._native_calls(response)
        return parse_tool_calls_detailed(response.message.content or "")

    def _native_calls(self, response: ChatResponse) -> ToolCallParseResult:
        return parse_native_tool_calls(response.message)

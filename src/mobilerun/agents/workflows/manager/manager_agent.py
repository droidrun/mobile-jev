"""
ManagerAgent - Planning and reasoning workflow.

This agent is responsible for:
- Analyzing the current state
- Creating plans and subgoals
- Tracking progress
- Deciding when tasks are complete
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Optional, Type, Unpack

from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun.config.prompts import PromptLoader
from mobilerun.core.state.base import _resize_model_screenshot_async
from mobilerun.logs.handlers import VERBOSE
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from opentelemetry import trace
from pydantic import BaseModel

from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.context.context_provider import (
    ContextProvider,
    update_context,
)
from mobilerun.agents.context.providers.app_card_context_provider import (
    context_provider_from_config,
)
from mobilerun.agents.llm.inference import acall_with_retries
from mobilerun.agents.llm.usage import get_usage_from_response
from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.telemetry.setup import record_langfuse_screenshot
from mobilerun.agents.utils.chat_utils import (
    _ensure_image_bytes,
    filter_empty_messages,
)
from mobilerun.agents.utils.prompt_resolver import PromptResolver
from mobilerun.agents.workflows import WorkflowOptions
from mobilerun.agents.workflows.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agents.workflows.droid.events import ExternalUserMessageAppliedEvent
from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.manager.contracts import ManagerToolContract
from mobilerun.agents.workflows.manager.events import (
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)
from mobilerun.agents.workflows.manager.prompts import (
    ManagerResponseValidationError,
    parse_manager_response,
    validate_manager_response,
)

if TYPE_CHECKING:
    from mobilerun.core.state.base import StateProvider

    from mobilerun.agents.config.user_config import (
        AgentConfig,
        TracingConfig,
    )
    from mobilerun.agents.tools.action_context import ActionContext
    from mobilerun.agents.tools.tool_registry import ToolRegistry


logger = logging.getLogger("mobilerun")


class ManagerAgent(Workflow, ManagerToolContract):
    """
    Planning and reasoning agent that decides what to do next.

    The Manager:
    1. Analyzes current device state and action history
    2. Creates plans with specific subgoals
    3. Tracks progress and completed steps
    4. Decides when tasks are complete or need to provide answers
    """

    def __init__(
        self,
        llm: LLM,
        action_ctx: "ActionContext | None",
        state_provider: "StateProvider | None",
        save_trajectory: str = "none",
        shared_state: MobileAgentState | None = None,
        agent_config: "AgentConfig | None" = None,
        registry: "ToolRegistry | None" = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: "TracingConfig | None" = None,
        context_provider: ContextProvider | None = None,
        **kwargs: Unpack[WorkflowOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        if agent_config is None:
            raise ValueError("Manager requires an agent configuration")
        self.config = agent_config.manager
        self.vision = self.config.vision
        self.action_ctx = action_ctx
        self.state_provider = state_provider
        self.save_trajectory = save_trajectory
        self._stream_screenshots = get_agent_settings().stream_screenshots
        self.shared_state = shared_state if shared_state is not None else MobileAgentState()
        self.registry = registry
        self.output_model = output_model
        self.agent_config = agent_config
        self.app_card_config = self.agent_config.app_cards
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config
        self.standard_tool_names: list[str] = []

        self.context_provider = (
            context_provider
            if context_provider is not None
            else context_provider_from_config(self.app_card_config)
        )

    async def _build_system_prompt(self) -> str:
        """Build system prompt with all context."""
        # Build error history if needed
        error_history = None
        if self.shared_state.error_flag_plan:
            k = self.shared_state.err_to_manager_thresh
            error_history = [
                {"action": act, "summary": summ, "error": err_des}
                for act, summ, err_des in zip(
                    self.shared_state.action_history[-k:],
                    self.shared_state.summary_history[-k:],
                    self.shared_state.error_descriptions[-k:],
                    strict=True,
                )
            ]

        # Output schema if provided
        output_schema = None
        if self.output_model is not None:
            output_schema = self.output_model.model_json_schema()

        # Build custom tools descriptions from registry.
        # Exclude standard tools (already described in the prompt template).
        custom_tools_descriptions = ""
        if self.registry:
            _standard = set(self.standard_tool_names)
            custom_tools_descriptions = self.registry.get_tool_descriptions_text(exclude=_standard)

        variables = {
            "instruction": self.shared_state.instruction,
            "device_date": self.shared_state.device_date,
            "extra_context": self.shared_state.extra_context,
            "important_notes": "",  # TODO: implement
            "error_history": error_history,
            "custom_tools_descriptions": custom_tools_descriptions,
            "variables": self.shared_state.custom_variables,
            "output_schema": output_schema,
            "platform": self.shared_state.platform,
        }

        custom_prompt = self.prompt_resolver.get_prompt("manager_system")
        if custom_prompt:
            return PromptLoader.render_template(custom_prompt, variables)
        else:
            return await PromptLoader.load_prompt(
                self.agent_config.get_manager_system_prompt_path(),
                variables,
            )

    def _build_user_message_content(self) -> str:
        """Build user message content with last action context."""
        parts = []

        # Add last thought
        if self.shared_state.last_thought:
            parts.append(f"<thought>\n{self.shared_state.last_thought}\n</thought>\n")

        # Add last action
        if self.shared_state.last_action:
            action_str = json.dumps(self.shared_state.last_action)
            parts.append(f"<last_action>\n{action_str}\n</last_action>\n")

        # Add last action summary
        if self.shared_state.last_summary:
            parts.append(
                f"<last_action_description>\n{self.shared_state.last_summary}\n</last_action_description>\n"
            )

        return "".join(parts)

    def _build_messages_with_context(
        self, system_prompt: str, screenshot: bytes | None = None
    ) -> list[ChatMessage]:
        """
        Build messages from history and inject current context.

        Args:
            system_prompt: System prompt text
            screenshot: Current screenshot if vision enabled

        Returns:
            List of ChatMessage objects ready for LLM
        """

        # Start with system message
        messages = [ChatMessage(role="system", content=system_prompt)]

        # Add accumulated message history (deep copy to avoid mutation)
        messages.extend(copy.deepcopy(self.shared_state.message_history))

        # Find last user message
        user_indices = [i for i, msg in enumerate(messages) if msg.role == MessageRole.USER]

        if user_indices:
            last_user_idx = user_indices[-1]

            # Add memory to last user message
            current_memory = (self.shared_state.agent_memory or "").strip()
            if current_memory:
                messages[last_user_idx].blocks.append(
                    TextBlock(text=f"\n<memory>\n{current_memory}\n</memory>\n")
                )

            # Add current device state
            current_state = self.shared_state.formatted_device_state.strip()
            if current_state:
                messages[last_user_idx].blocks.append(
                    TextBlock(text=f"\n<device_state>\n{current_state}\n</device_state>\n")
                )

            # Add screenshot if vision enabled
            if screenshot and self.vision:
                messages[last_user_idx].blocks.append(ImageBlock(image=screenshot))

            # Add previous device state to second-to-last user message
            if len(user_indices) >= 2:
                second_last_idx = user_indices[-2]
                prev_state = self.shared_state.previous_formatted_device_state.strip()
                if prev_state:
                    messages[second_last_idx].blocks.append(
                        TextBlock(
                            text=f"\n<previous_device_state>\n{prev_state}\n</previous_device_state>\n"
                        )
                    )

        messages = filter_empty_messages(messages)
        return messages

    async def _validate_and_retry(self, messages: list[ChatMessage], initial_response: str) -> str:
        """Validate LLM response and retry if needed."""
        output = initial_response
        parsed = parse_manager_response(output)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            validation = validate_manager_response(parsed)
            if validation.is_valid:
                return output

            validation_error = validation.error_message or "Invalid manager response."
            error_message = f"{validation_error}\nRetry again."
            retry_count += 1
            logger.warning(
                f"Manager response invalid (retry {retry_count}/{max_retries}): {error_message}"
            )

            retry_messages = messages + [
                ChatMessage(role="assistant", content=output),
                ChatMessage(role="user", content=error_message),
            ]

            try:
                response = await acall_with_retries(
                    self.llm, retry_messages, stream=self.agent_config.streaming
                )
                output = response.message.content
                if output is None:
                    raise ValueError("Manager LLM returned no text content")
                parsed = parse_manager_response(output)
            except Exception as e:
                logger.error(f"LLM retry failed: {describe_error(e)}")
                raise ManagerResponseValidationError(validation_error) from e

        validation = validate_manager_response(parsed)
        if validation.is_valid:
            return output
        raise ManagerResponseValidationError(
            validation.error_message or "Invalid manager response."
        )

    # ========================================================================
    # Workflow Steps
    # ========================================================================

    @step
    async def prepare_context(self, ctx: Context, ev: StartEvent) -> ManagerContextEvent:
        """Gather context and prepare manager prompt."""
        logger.debug(" Preparing manager context...")
        if self.action_ctx is None or self.state_provider is None:
            raise RuntimeError("Manager requires an action context and state provider")

        # Capture screenshot if needed
        screenshot = None
        if self.vision or self._stream_screenshots or self.save_trajectory != "none":
            try:
                screenshot = await self.action_ctx.driver.screenshot()

                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=parent_span,
                        screenshots_enabled=bool(
                            self.tracing_config and self.tracing_config.langfuse_screenshots
                        ),
                        vision_enabled=self.vision,
                    )
                    logger.debug(" Screenshot captured for Manager")
            except DeviceDisconnectedError:
                raise
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        # Get and format device state
        ui_state = await self.state_provider.get_state()
        self.action_ctx.ui = ui_state

        # Update shared state (previous ← current, current ← new)
        self.shared_state.previous_formatted_device_state = self.shared_state.formatted_device_state
        self.shared_state.formatted_device_state = ui_state.formatted_text
        self.shared_state.focused_text = ui_state.focused_text
        self.shared_state.a11y_tree = ui_state.elements
        self.shared_state.phone_state = ui_state.phone_state

        # Update package/activity tracking
        self.shared_state.update_current_app(
            package_name=ui_state.phone_state.get("packageName", "Unknown"),
            activity_name=ui_state.phone_state.get("currentApp", "Unknown"),
        )

        # Stream UI state for trajectory
        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=ui_state.elements))

        await update_context(self.context_provider, self.shared_state, screenshot)

        self.shared_state.screenshot = screenshot

        # Build user message and add to history
        user_content = self._build_user_message_content()

        drained = self.shared_state.drain_user_messages()
        if drained:
            external_block = "\n".join(
                f"<external_user_message>\n{m.message}\n</external_user_message>" for m in drained
            )
            user_content += "\n" + external_block + "\n"
            logger.log(
                VERBOSE,
                f" Applied {len(drained)} external user message(s)",
                extra={"color": "cyan"},
            )
            ctx.write_event_to_stream(
                ExternalUserMessageAppliedEvent(
                    message_ids=[m.id for m in drained],
                    consumer="manager",
                    step_number=self.shared_state.step_number,
                )
            )

        self.shared_state.message_history.append(ChatMessage(role="user", content=user_content))

        event = ManagerContextEvent()
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(self, ctx: Context, ev: ManagerContextEvent) -> ManagerResponseEvent:
        """Get LLM response."""
        logger.debug(" Manager thinking about the plan...")

        source = self.shared_state.screenshot
        screenshot = _ensure_image_bytes(source) if source and self.vision else None

        # Build system prompt
        system_prompt = await self._build_system_prompt()

        if screenshot and self.vision:
            if self.state_provider is None:
                raise RuntimeError("Manager requires a state provider for vision")
            screenshot = await _resize_model_screenshot_async(self.state_provider, screenshot)

        # Build messages with context
        messages = self._build_messages_with_context(
            system_prompt=system_prompt, screenshot=screenshot
        )

        try:
            logger.log(VERBOSE, " Manager response:", extra={"color": "cyan"})
            response = await acall_with_retries(
                self.llm, messages, stream=self.agent_config.streaming
            )
            output = response.message.content
            if output is None:
                raise ValueError("Manager LLM returned no text content")
        except Exception as e:
            error = describe_error(e)
            logger.error(f"LLM call failed: {error}")
            raise RuntimeError(f"Error calling LLM in manager: {error}") from e

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        output = await self._validate_and_retry(messages, output)

        event = ManagerResponseEvent(response=output, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ManagerResponseEvent
    ) -> ManagerPlanDetailsEvent:
        """Parse LLM response and update state."""
        logger.debug(" Processing manager response...")

        output = ev.response
        parsed = parse_manager_response(output)

        # Update memory (append)
        memory_update = parsed.get("memory", "").strip()
        if memory_update:
            self.shared_state.append_memory(memory_update)

        # Append assistant response to message history
        self.shared_state.message_history.append(ChatMessage(role="assistant", content=output))

        # Update unified state fields
        self.shared_state.previous_plan = self.shared_state.plan
        self.shared_state.plan = parsed["plan"]
        self.shared_state.current_subgoal = parsed["current_subgoal"]
        self.shared_state.last_thought = parsed["thought"]
        self.shared_state.answer = parsed["answer"]

        if parsed.get("progress_summary"):
            self.shared_state.progress_summary = parsed["progress_summary"]

        event = ManagerPlanDetailsEvent(
            plan=parsed["plan"],
            subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            answer=parsed["answer"],
            memory_update=memory_update,
            progress_summary=parsed.get("progress_summary", ""),
            success=parsed["success"],
            full_response=output,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: ManagerPlanDetailsEvent) -> StopEvent:
        logger.debug(" Manager planning complete")

        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.subgoal,
                "thought": ev.thought,
                "answer": ev.answer,
                "memory_update": ev.memory_update,
                "success": ev.success,
            }
        )

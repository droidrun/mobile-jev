"""
ExecutorAgent - Action execution workflow.

This agent is responsible for:
- Taking a specific subgoal from the Manager
- Analyzing the current UI state
- Selecting and executing appropriate actions
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional, Unpack

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun.config.prompts import PromptLoader
from mobilerun.core.state.base import _resize_model_screenshot_async
from mobilerun.logs.handlers import VERBOSE

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.llm.inference import acall_with_retries
from mobilerun.agents.llm.usage import get_usage_from_response
from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.utils.chat_utils import _ensure_image_bytes
from mobilerun.agents.utils.prompt_resolver import PromptResolver
from mobilerun.agents.workflows import WorkflowOptions
from mobilerun.agents.workflows.executor.events import (
    ExecutorActionEvent,
    ExecutorActionResultEvent,
    ExecutorContextEvent,
    ExecutorResponseEvent,
)
from mobilerun.agents.workflows.executor.prompts import parse_executor_response

if TYPE_CHECKING:
    from mobilerun.agents.tools.action_context import ActionContext
    from mobilerun.agents.tools.tool_registry import ToolRegistry
    from mobilerun.agents.workflows.droid import MobileAgentState

logger = logging.getLogger("mobilerun")


class ExecutorAgent(Workflow):
    """
    Action execution agent that performs specific actions.

    Single-turn agent: receives subgoal, selects action, executes it.
    Uses ChatMessage objects directly for LLM calls.
    """

    # Flow-control tools hidden from executor's LLM prompt
    _EXCLUDE_TOOLS = {"complete"}

    def __init__(
        self,
        llm: LLM,
        registry: "ToolRegistry | None",
        action_ctx: "ActionContext | None",
        shared_state: "MobileAgentState",
        agent_config: AgentConfig,
        prompt_resolver: Optional[PromptResolver] = None,
        **kwargs: Unpack[WorkflowOptions],
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.agent_config = agent_config
        self.config = agent_config.executor
        self.vision = agent_config.executor.vision
        self.registry = registry
        self.action_ctx = action_ctx
        self.shared_state = shared_state
        self.prompt_resolver = prompt_resolver or PromptResolver()

        logger.debug("ExecutorAgent initialized.")

    @step
    async def prepare_context(self, ctx: Context, ev: StartEvent) -> ExecutorContextEvent:
        """Prepare executor context and prompt."""
        if self.registry is None:
            raise RuntimeError("Executor requires a tool registry")
        subgoal = ev.get("subgoal", "")
        logger.debug(f" Executor thinking about action for: {subgoal}")

        # Build action history (last 5)
        action_history = []
        if self.shared_state.action_history:
            n = min(5, len(self.shared_state.action_history))
            action_history = [
                {"action": act, "summary": summ, "outcome": outcome, "error": err}
                for act, summ, outcome, err in zip(
                    self.shared_state.action_history[-n:],
                    self.shared_state.summary_history[-n:],
                    self.shared_state.action_outcomes[-n:],
                    self.shared_state.error_descriptions[-n:],
                    strict=True,
                )
            ]

        # Build prompt variables
        variables = {
            "instruction": self.shared_state.instruction,
            "extra_context": self.shared_state.extra_context,
            "device_state": self.shared_state.formatted_device_state,
            "plan": self.shared_state.plan,
            "subgoal": subgoal,
            "progress_status": self.shared_state.progress_summary,
            "atomic_actions": self.registry.get_signatures(exclude=self._EXCLUDE_TOOLS),
            "action_history": action_history,
            "variables": self.shared_state.custom_variables,
            "platform": self.shared_state.platform,
        }

        custom_prompt = self.prompt_resolver.get_prompt("executor_system")
        if custom_prompt:
            prompt_text = PromptLoader.render_template(custom_prompt, variables)
        else:
            prompt_text = await PromptLoader.load_prompt(
                self.agent_config.get_executor_system_prompt_path(),
                variables,
            )

        # Build message
        messages = [ChatMessage(role="user", blocks=[TextBlock(text=prompt_text)])]

        # Add screenshot if vision enabled
        if self.vision:
            screenshot = self.shared_state.screenshot
            if screenshot is not None:
                if self.action_ctx is None:
                    raise RuntimeError("Executor requires an action context for vision")
                screenshot = await _resize_model_screenshot_async(
                    self.action_ctx.state_provider, _ensure_image_bytes(screenshot)
                )
                messages[0].blocks.append(ImageBlock(image=screenshot))
                logger.debug(" Using screenshot for Executor")
            else:
                logger.warning(" Vision enabled but no screenshot available")
        await ctx.store.set("executor_messages", messages)
        event = ExecutorContextEvent(subgoal=subgoal)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def get_response(self, ctx: Context, ev: ExecutorContextEvent) -> ExecutorResponseEvent:
        """Get LLM response."""
        logger.debug("Executor getting LLM response...")

        # Get messages from context
        messages = await ctx.store.get("executor_messages")

        try:
            logger.log(VERBOSE, "Executor response:", extra={"color": "green"})
            response = await acall_with_retries(
                self.llm,
                messages,
                stream=self.agent_config.streaming,
                log_response=not (self.registry and self.registry.has_sensitive_actions()),
            )
            response_text = str(response)
        except ValueError as e:
            logger.warning(f"Executor LLM returned empty response: {e}")
            error_response = (
                "### Thought\nExecutor failed to respond, try again\n"
                '### Action\n{"action": "invalid"}\n'
                "### Description\nExecutor failed to respond, try again"
            )
            event = ExecutorResponseEvent(response=error_response, usage=None)
            ctx.write_event_to_stream(event)
            return event
        except Exception as e:
            raise RuntimeError(f"Error calling LLM in executor: {describe_error(e)}") from e

        # Extract usage
        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        await ctx.store.set("executor_raw_response", response_text)
        safe_response = response_text
        try:
            parsed_action = json.loads(parse_executor_response(response_text)["action"])
            action_name = parsed_action.pop("action", "")
            if self.registry:
                safe_response = self.registry.redact_text(action_name, parsed_action, response_text)
        except (ValueError, KeyError, TypeError):
            if self.registry and self.registry.has_sensitive_actions():
                safe_response = "[Unparseable response withheld]"
        event = ExecutorResponseEvent(response=safe_response, usage=usage)
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ExecutorResponseEvent
    ) -> ExecutorActionEvent:
        """Parse LLM response and extract action."""
        logger.debug(" Processing executor response...")

        response_text = await ctx.store.get("executor_raw_response", ev.response)

        try:
            parsed = parse_executor_response(response_text)
        except Exception:
            logger.error("Failed to parse executor response")
            return ExecutorActionEvent(
                action_json=json.dumps({"action": "invalid"}),
                thought="Failed to parse response",
                description="Invalid response format from LLM",
                full_response=ev.response,
            )

        # Update unified state
        try:
            action = json.loads(parsed["action"])
            if not isinstance(action, dict):
                raise ValueError("Action must be an object")
        except (TypeError, ValueError):
            action = {"action": "invalid"}
            parsed["action"] = json.dumps(action)
            parsed["thought"] = "Invalid action"
            parsed["description"] = "Invalid action"
        action_name = action.get("action", "")
        safe_args = self.registry.redact_args(action_name, action) if self.registry else action
        safe_thought = (
            self.registry.redact_text(action_name, action, parsed["thought"])
            if self.registry
            else parsed["thought"]
        )
        safe_description = (
            self.registry.redact_text(action_name, action, parsed["description"])
            if self.registry
            else parsed["description"]
        )
        self.shared_state.last_thought = safe_thought
        await ctx.store.set("executor_raw_action", parsed["action"])
        event = ExecutorActionEvent(
            action_json=json.dumps(safe_args),
            thought=safe_thought,
            description=safe_description,
            full_response=ev.response,
        )

        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute(self, ctx: Context, ev: ExecutorActionEvent) -> ExecutorActionResultEvent:
        """Execute the action."""
        logger.debug(f" Executing action: {ev.description}")

        try:
            action_dict = json.loads(await ctx.store.get("executor_raw_action", ev.action_json))
        except json.JSONDecodeError as e:
            logger.error(f" Failed to parse action JSON: {e}")
            event = ExecutorActionResultEvent(
                action={"action": "invalid"},
                success=False,
                error=f"Invalid action JSON: {str(e)}",
                summary="Failed to parse action",
                thought=ev.thought,
                full_response=ev.full_response,
            )
            ctx.write_event_to_stream(event)
            return event

        # Extract action type and args, dispatch via registry
        action_type = action_dict.get("action", "unknown")
        action_args = {k: v for k, v in action_dict.items() if k != "action"}

        if self.registry is None or self.action_ctx is None:
            raise RuntimeError("Executor requires a tool registry and action context")
        result = await self.registry.execute(
            action_type, action_args, self.action_ctx, workflow_ctx=ctx
        )
        safe_action = {
            "action": action_type,
            **self.registry.redact_args(action_type, action_args),
        }

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.debug(f"{'' if result.success else ''} Execution complete: {result.summary}")

        event = ExecutorActionResultEvent(
            action=safe_action,
            success=result.success,
            error="" if result.success else result.summary,
            summary=result.summary,
            thought=ev.thought,
            full_response=ev.full_response,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def finalize(self, ctx: Context, ev: ExecutorActionResultEvent) -> StopEvent:
        """Return executor results to parent workflow."""
        logger.debug(" Executor execution complete")

        return StopEvent(
            result={
                "action": ev.action,
                "outcome": ev.success,
                "error": ev.error,
                "summary": ev.summary,
                "thought": ev.thought,
            }
        )

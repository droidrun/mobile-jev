"""Own the explicit handoff option and route retries for that step."""

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from workflows import Context

from mobilerun_agents.agent.tools.atomic import AtomicTool
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.prompts import render_prompt

from .base import ActionSelector, Selection
from .decision_model.base import DecisionSelector, requires_typed_completion
from .llm.calls import InvalidResponse


class DelegatingSelector(ActionSelector):
    def __init__(
        self,
        primary: DecisionSelector,
        delegate: ActionSelector,
    ) -> None:
        self.primary, self.delegate = primary, delegate
        self._extra = (
            AtomicTool("delegate", label=render_prompt("assets/prompts/delegate/option.jinja2")),
        )

    async def select(
        self, ctx: Context[AgentState], device: Device, memory: Memory, registry: ToolRegistry
    ) -> Selection:
        state = await ctx.store.get_state()
        if state.current is None:
            raise ValueError("Selection requires a current observation")
        if state.retry_with_delegate and state.repairs:
            return await self._select_delegate(ctx, device, memory, registry)

        selection = await self.primary.select(
            ctx, device, memory, registry, extra=self._extra, intent=state.delegate_intent
        )
        if selection.abstained:
            return selection
        if any(call.tool_name == "delegate" for call in selection.calls):
            if len(selection.calls) != 1 or selection.calls[0].tool_kwargs:
                raise InvalidResponse(
                    render_prompt("assets/prompts/delegate/invalid.jinja2", kind="parameters")
                )
        elif not (
            requires_typed_completion(registry)
            and any(
                call.tool_name == "complete" and call.tool_kwargs.get("success") is True
                for call in selection.calls
            )
        ):
            return selection
        async with ctx.store.edit_state() as state:
            state.retry_with_delegate = True
        return await self._select_delegate(ctx, device, memory, registry)

    async def _select_delegate(
        self,
        ctx: Context[AgentState],
        device: Device,
        memory: Memory,
        registry: ToolRegistry,
    ) -> Selection:
        selection = await self.delegate.select(ctx, device, memory, registry)
        if any(call.tool_name == "delegate" for call in selection.calls):
            raise InvalidResponse(
                render_prompt("assets/prompts/delegate/invalid.jinja2", kind="recursive")
            )
        if selection.thought and selection.thought.strip():
            async with ctx.store.edit_state() as state:
                state.delegate_intent = selection.thought.strip()
        return selection

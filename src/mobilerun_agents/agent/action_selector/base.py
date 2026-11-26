"""Selectors choose actions; the workflow alone executes them."""

import asyncio
from abc import ABC, abstractmethod
from typing import Literal

from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.devices import Device
from pydantic import BaseModel, Field
from workflows import Context

from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState, JevAbstention

NativeToolAPI = Literal["responses", "chat_completions", "anthropic", "gemini"]


class Selection(BaseModel):
    calls: list[ToolSelection] = Field(default_factory=list)
    thought: str | None = None
    confidence: float | None = None
    abstained: bool = False
    jev_abstention: JevAbstention | None = None


class ActionSelector(ABC):
    @abstractmethod
    async def select(
        self, ctx: Context[AgentState], device: Device, memory: Memory, registry: ToolRegistry
    ) -> Selection: ...


async def screenshot_bytes(
    ctx: Context[AgentState], snapshot: StepSnapshot, device: Device
) -> bytes | None:
    screenshot = snapshot.screenshot
    if screenshot is None and snapshot.ui.elements:
        async with ctx.store.edit_state() as state:
            assert state.current is not None
            await state.current.capture_screenshot(device)
            screenshot = state.current.screenshot
    if screenshot is None:
        return None

    def read() -> bytes:
        assert screenshot is not None
        with screenshot.resolve_image() as stream:
            return bytes(stream.read())

    return await asyncio.to_thread(read)


def previous_action_feedback(state: AgentState) -> str | None:
    changed = state.actions[-1].screen_changed if state.actions else None
    return {
        True: "Previous action changed the screen",
        False: "Previous action did not change the screen",
        None: None,
    }[changed]

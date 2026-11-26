from contextlib import asynccontextmanager
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llama_index.core.llms import ImageBlock
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported

from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.tools.registry import ToolRegistry, build_registry
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import ToolsConfig
from tests.unit.device_fixtures import MemoryDevice


class Store:
    def __init__(self, get_state):
        self.get_state = get_state

    @asynccontextmanager
    async def edit_state(self):
        yield await self.get_state()


class Turn:
    def __init__(self, goal="Test", tools=(), screenshot=None, elements=None, device=None, ui=None):
        self.device = device or MemoryDevice()
        try:
            _ = self.device.state._resolved_provider
        except DeviceOperationNotSupported:
            self.device._set_state_provider(StateProvider(self.device))
        self.state = AgentState(goal=goal)
        self.observe(ui or UIState(elements or [], "screen", "", {}, 100, 200), screenshot)
        self.events = []
        self.ctx = SimpleNamespace(
            store=Store(self.get_state),
            write_event_to_stream=self.events.append,
        )
        self.memory = SimpleNamespace(aget=AsyncMock(return_value=[]))
        self.registry = (
            ToolRegistry(self.device, _members=tuple(tools))
            if tools
            else build_registry(self.device, [], ToolsConfig())
        )

    def observe(self, ui, screenshot=None):
        self.state.current = StepSnapshot(
            ui=ui, screenshot=ImageBlock(image=screenshot) if screenshot else None
        )
        self.state.repairs = 0
        self.state.retry_with_delegate = False

    async def get_state(self):
        return self.state

    async def select(self, selector, stream=False, parallel_tools=True):
        if isinstance(selector, LLMSelector):
            selector = copy(selector)
            selector.config = selector.config.model_copy(
                update={"streaming": stream, "parallel_tools": parallel_tools}
            )
        return await selector.select(self.ctx, self.device, self.memory, self.registry)

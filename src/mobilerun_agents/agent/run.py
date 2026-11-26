"""Run-local state available to every decorated tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.macro.recorder import MacroRecorder
from pydantic import BaseModel

from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import AgentConfig, ToolsConfig

if TYPE_CHECKING:
    from mobilerun_agents.agent.action_selector.base import ActionSelector

    from .memory.snapshot import StepSnapshot
    from .tools.registry import ToolRegistry


@dataclass(frozen=True)
class InstalledApp:
    identifier: str
    label: str


@dataclass
class Capture:
    image: bytes | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(kw_only=True)
class Run:
    device: Device
    ui: UIState = field(init=False, repr=False)
    tool_config: ToolsConfig = field(default_factory=ToolsConfig)
    macro_recorder: MacroRecorder | None = field(default_factory=MacroRecorder)

    snapshot: StepSnapshot | None = field(default=None, init=False)
    capture: Capture = field(default_factory=Capture, init=False)
    registry: ToolRegistry = field(init=False)
    memory: Memory
    config: AgentConfig
    output: type[BaseModel] | None = None
    action_selector: ActionSelector | None = None
    _apps: tuple[InstalledApp, ...] | None = field(default=None, init=False, repr=False)
    atomic_drops: set[str] = field(default_factory=set, init=False, repr=False)
    _state_snapshot: AgentState = field(default_factory=AgentState, init=False, repr=False)

    @property
    def state(self) -> AgentState:
        """Detached selection state; changes to this value never enter the workflow store."""
        return self._state_snapshot.model_copy(deep=True)

    @property
    def selector(self) -> ActionSelector:
        assert self.action_selector is not None
        return self.action_selector

    @property
    def llm(self) -> LLM | None:
        return self.selector.shared_llm

    async def installed_apps(self) -> tuple[InstalledApp, ...]:
        if self._apps is None:
            try:
                records = await self.device.list_apps(include_system=True)
            except DeviceOperationNotSupported:
                records = []
            self._apps = tuple(
                InstalledApp(identifier, record.get("label") or identifier)
                for record in records
                if (identifier := record.get("package_name") or record.get("package"))
            )
        return self._apps

    _memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def remember(self, message: ChatMessage) -> None:
        async with self._memory_lock:
            await self.memory.aput(message)

    async def history(self) -> list[ChatMessage]:
        async with self._memory_lock:
            return await self.memory.aget(snapshot=self.snapshot)

    async def screenshot(self) -> bytes:
        async with self.capture.lock:
            if self.capture.image is None:
                self.capture.image = await self.device.screenshot()
            return self.capture.image

    @property
    def state_provider(self) -> StateProvider:
        # UIStateView exposes capabilities, but not the provider's coordinate flags.
        # Read those flags only; never replace the caller's provider.
        return self.device.state._resolved_provider

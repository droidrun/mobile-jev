"""Run-local state available to every decorated tool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llama_index.core.llms import ChatMessage, ImageBlock
from llama_index.core.memory import Memory
from llama_index.core.tools import BaseTool
from mobilerun.core.devices import Device
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.macro.recorder import MacroRecorder
from pydantic import BaseModel

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import AgentConfig, ToolsConfig

if TYPE_CHECKING:
    from .tools.helpers.images import ImageStore
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
    macro_recorder: MacroRecorder | None = None

    snapshot: StepSnapshot | None = field(default=None, init=False)
    images: ImageStore | None = None
    capture: Capture = field(default_factory=Capture, init=False)
    registry: ToolRegistry = field(init=False)
    memory: Memory
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    config: AgentConfig
    output: type[BaseModel] | None = None
    _apps: tuple[InstalledApp, ...] | None = field(default=None, init=False, repr=False)
    atomic_drops: set[str] = field(default_factory=set, init=False, repr=False)
    selection_exclude: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    selection_include: tuple[BaseTool, ...] = field(default=(), init=False, repr=False)
    avoid_signature: str = field(default="", init=False, repr=False)
    _state_snapshot: AgentState = field(default_factory=AgentState, init=False, repr=False)

    @property
    def state(self) -> AgentState:
        """Detached selection state; changes to this value never enter the workflow store."""
        return self._state_snapshot.model_copy(deep=True)

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

    async def history(self) -> list[ChatMessage]:
        async with self.memory_lock:
            if self.snapshot is not None:
                self.snapshot._actions = tuple(self._state_snapshot.actions)
                self.snapshot._thresholds = (
                    self.config.stuck_exclude_after,
                    self.config.no_progress_note_after,
                    self.config.cycle_note_after,
                )
            try:
                return await self.memory.aget(snapshot=self.snapshot)
            finally:
                if self.snapshot is not None:
                    self.snapshot._actions = ()

    async def screenshot(self) -> bytes:
        async with self.capture.lock:
            if self.capture.image is None:
                self.capture.image = await self.device.screenshot()
            return self.capture.image

    def save_screenshot(self) -> None:
        if self.snapshot is not None and self.images is not None and self.capture.image:
            self.snapshot.attach_screenshot(ImageBlock(path=self.images.save(self.capture.image)))

    @property
    def state_provider(self) -> StateProvider:
        # UIStateView exposes capabilities, but not the provider's coordinate flags.
        # Read those flags only; never replace the caller's provider.
        return self.device.state._resolved_provider

import asyncio
from pathlib import Path
from typing import Literal

from pydantic import PrivateAttr, TypeAdapter, ValidationError

from mobilerun_agents.agent.workflows.core.state import AgentState

from . import StateBlock

_MAPPING = TypeAdapter(dict[str, str])


class AppCardBlock(StateBlock[AgentState, str]):
    """Load app-specific guidance from the local card root for the current platform."""

    name: str = "app_cards"
    priority: int = 1
    platform: Literal["android", "ios"]
    cards_dir: Path
    _cache: dict[str, str] = PrivateAttr(default_factory=dict)

    def _local_card(self, folder: Path, package: str) -> str:
        try:
            mapping = _MAPPING.validate_json((folder / "app_cards.json").read_bytes())
            if package not in mapping:
                return ""
            path = (folder / mapping[package]).resolve()
            if not path.is_relative_to(folder.resolve()):
                return ""
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, ValidationError):
            return ""

    async def render(self, state: AgentState) -> str:
        snapshot = state.current
        if snapshot is None or not (package := snapshot.ui.phone_state.get("packageName")):
            return ""
        if package not in self._cache:
            self._cache[package] = await asyncio.to_thread(
                self._local_card, self.cards_dir / self.platform, package
            )
        return self._cache[package]

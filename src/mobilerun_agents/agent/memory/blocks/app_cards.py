import asyncio
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, PrivateAttr, TypeAdapter, ValidationError

from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.memory.snapshot import StepSnapshot

_MAPPING = TypeAdapter(dict[str, str])


class _CardResponse(BaseModel):
    app_card: str


class AppCardBlock(SnapshotBlock):
    """Inject a base-URL HTTP client for server lookup, a card root for local fallback, or both."""

    name: str = "app_card"
    priority: int = 2
    platform: Literal["android", "ios"]
    cards_dir: Path | None = None
    client: httpx.AsyncClient | None = None
    _cache: dict[tuple[str, str], str] = PrivateAttr(default_factory=dict)

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

    async def render(self, snapshot: StepSnapshot) -> str:
        if not (package := snapshot.package):
            return ""
        key = package, snapshot.goal
        if key in self._cache:
            return self._cache[key]
        card = ""
        cacheable = True
        if self.client is not None:
            try:
                response = await self.client.post(
                    "app-cards",
                    json={
                        "package_name": package,
                        "instruction": snapshot.goal,
                        "platform": self.platform,
                    },
                )
                if response.status_code != 404:
                    response.raise_for_status()
                    card = _CardResponse.model_validate_json(response.content).app_card
            except (httpx.HTTPError, ValidationError):
                cacheable = False  # retry on the next observation
        if not card and self.cards_dir is not None:
            card = await asyncio.to_thread(
                self._local_card, self.cards_dir / self.platform, package
            )
        if cacheable:
            self._cache[key] = card
        return card

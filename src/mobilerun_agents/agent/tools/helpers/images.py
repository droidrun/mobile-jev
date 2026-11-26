"""Store one on-disk image per screenshot captured in a run."""

import shutil
from collections.abc import Collection
from pathlib import Path

from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.settings import get_agent_settings


def _contains(value: object, protected: str) -> bool:
    match value:
        case dict():
            return any(_contains(k, protected) or _contains(v, protected) for k, v in value.items())
        case list() | tuple():
            return any(_contains(item, protected) for item in value)
        case str():
            return protected in value
        case int() | float() | bool():
            return protected == str(value)
        case _:
            return False


class ImageStore:
    def __init__(self, run_id: str, keep: bool) -> None:
        if run_id in {"", ".", ".."} or Path(run_id).name != run_id:
            raise ValueError("run_id must be a single directory name")
        self.folder = get_agent_settings().images_dir / run_id
        self._keep = keep
        self._sequence = 0

    def save(self, data: bytes) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        while True:
            self._sequence += 1
            path = self.folder / f"{self._sequence:04}.png"
            try:
                file = path.open("xb")
            except FileExistsError:
                continue
            with file:
                file.write(data)
            return path

    def forget(self, state: AgentState, step: int, protected_values: Collection[str]) -> None:
        """Delete prior image files and replace their state references at action commit."""
        if not protected_values:
            return
        records = [
            (record.step, record.model_dump(exclude={"before": {"screenshot"}}))
            for record in state.actions
        ]
        first = min(
            next((number for number, raw in records if _contains(raw, value)), step - 1)
            for value in protected_values
        )
        for record in state.actions:
            if first <= record.step <= step and record.before.screenshot is not None:
                path = record.before.screenshot.path
                if path is not None:
                    path.unlink(missing_ok=True)
                record.before = record.before.model_copy(update={"screenshot": None})

    def close(self) -> None:
        if not self._keep and self.folder.exists():
            shutil.rmtree(self.folder)

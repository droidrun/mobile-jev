"""Store one on-disk image per screenshot captured in a run."""

import shutil
from pathlib import Path

from mobilerun_agents.config.settings import get_agent_settings


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

    def close(self) -> None:
        if not self._keep and self.folder.exists():
            shutil.rmtree(self.folder)

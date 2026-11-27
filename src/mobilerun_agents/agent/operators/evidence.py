"""Write the add-on's reasoning beside, but never into, core action rows."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

from mobilerun_agents.agent.workflows.core.privacy import mask_value

AgentEntryKind = Literal["thought", "plan", "memory", "tool"]


class AgentEntry(BaseModel):
    kind: AgentEntryKind
    text: str
    step_ids: list[int] = Field(default_factory=list)


_entries = TypeAdapter(list[AgentEntry])


class AgentEvidence:
    """Atomically replace the JSON array so readers never see a partial update.

    Writes scale with the small, step-budgeted reasoning history. Reading back
    the array also preserves entries from previous agents sharing this run.
    """

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "agent.json"

    def append(self, entry: AgentEntry) -> None:
        try:
            history = _entries.validate_json(self._path.read_bytes())
        except FileNotFoundError:
            history = []
        history.append(entry.model_copy(update={"text": mask_value(entry.text)}))
        temporary = None
        try:
            with NamedTemporaryFile(dir=self._path.parent, delete=False) as file:
                temporary = Path(file.name)
                file.write(_entries.dump_json(history) + b"\n")
            os.replace(temporary, self._path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

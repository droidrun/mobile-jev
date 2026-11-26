from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.memory.snapshot import StepSnapshot


class RunNotesBlock(SnapshotBlock):
    """Explicit facts only: no model inference or copying raw tool arguments."""

    name: str = "run_notes"
    priority: int = 3

    async def render(self, snapshot: StepSnapshot) -> str:
        return "\n".join(snapshot.notes)

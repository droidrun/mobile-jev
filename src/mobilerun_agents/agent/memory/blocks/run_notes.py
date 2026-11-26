from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot


class RunNotesBlock(SnapshotBlock):
    """Explicit facts only: no model inference or copying raw tool arguments."""

    name: str = "run_notes"
    priority: int = 0

    async def render(self, snapshot: StepSnapshot) -> str:
        return "\n".join([f"Goal: {snapshot.goal}", *snapshot.notes])

from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.memory.snapshot import StepSnapshot


class DeviceContextBlock(SnapshotBlock):
    name: str = "device_context"
    priority: int = 4

    async def render(self, snapshot: StepSnapshot) -> str:
        if not snapshot.device_context:
            return ""
        return snapshot.model_dump_json(include={"device_context"})

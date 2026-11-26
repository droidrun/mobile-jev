from llama_index.core.base.llms.types import ChatMessage

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot

from . import ObservationBlock


class DeviceContextBlock(ObservationBlock):
    name: str = "device_context"
    priority: int = 0

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        snapshot: StepSnapshot | None = None,
        **kwargs: object,
    ) -> str:
        if snapshot is None or not snapshot.device_context:
            return ""
        return snapshot.model_dump_json(include={"device_context"})

import asyncio
from unittest.mock import Mock

from mobilerun.core.devices import Device
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from mobilerun_agents.agent.memory.memory import build_executor_memory
from mobilerun_agents.agent.workflows.resources import (
    DeviceResource,
    MemoryResource,
    Services,
    ServicesResource,
    run_resources,
)
from mobilerun_agents.config.user_config import MemoryConfig


class Again(Event):
    pass


class Probe(Workflow):
    @step
    async def start(
        self,
        ev: StartEvent,
        device: DeviceResource,
        memory: MemoryResource,
        services: ServicesResource,
    ) -> Again:
        self.first = (device, memory, services)
        return Again()

    @step
    async def end(
        self, ev: Again, device: DeviceResource, memory: MemoryResource, services: ServicesResource
    ) -> StopEvent:
        assert all(
            actual is expected
            for actual, expected in zip((device, memory, services), self.first, strict=True)
        )
        return StopEvent(result=memory.session_id)


def test_prebuilt_resources_are_injected_per_workflow_and_child():
    async def run():
        device = Mock(spec=Device)
        first = build_executor_memory(MemoryConfig(), "android")
        child = build_executor_memory(MemoryConfig(), "android")
        registry, images = Mock(), Mock()
        parent_services = Services(registry=registry, images=images)
        child_services = Services(registry=registry, images=images)
        parent_flow = Probe(resource_manager=await run_resources(device, first, parent_services))
        child_flow = Probe(resource_manager=await run_resources(device, child, child_services))
        assert await asyncio.gather(parent_flow.run(), child_flow.run()) == [
            first.session_id,
            child.session_id,
        ]
        assert first is not child and first.session_id != child.session_id
        assert parent_flow.first[0] is child_flow.first[0] is device
        assert parent_flow.first[1] is first and child_flow.first[1] is child
        assert parent_flow.first[2] is parent_services and child_flow.first[2] is child_services

    asyncio.run(run())

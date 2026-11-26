"""Inject prebuilt live inputs through LlamaIndex's ResourceManager.

The operator builds its inputs and passes `resource_manager=await
run_resources(device, memory, services)` to the workflow constructor, which
forwards it to `Workflow`. ResourceManager.set binds the cached descriptors by
name before steps run; no factory needs access to context or ambient state.
Construct a new workflow and manager for every run and planning child. A child
gets its own memory; the phone and image store can be passed on explicitly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from workflows.resource import Resource, ResourceManager

if TYPE_CHECKING:
    from mobilerun_agents.agent.tools.helpers.images import ImageStore
    from mobilerun_agents.agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class Services:
    registry: "ToolRegistry"
    images: "ImageStore"


def _device() -> Device:
    raise RuntimeError("Bind the run's device with run_resources()")


def _memory() -> Memory:
    raise RuntimeError("Bind the run's memory with run_resources()")


def _services() -> Services:
    raise RuntimeError("Bind the run's services with run_resources()")


_device_resource = Resource(_device)
_memory_resource = Resource(_memory)
_services_resource = Resource(_services)

DeviceResource = Annotated[Device, _device_resource]
MemoryResource = Annotated[Memory, _memory_resource]
ServicesResource = Annotated[Services, _services_resource]


async def run_resources(device: Device, memory: Memory, services: Services) -> ResourceManager:
    manager = ResourceManager()
    await manager.set(_device_resource.name, device)
    await manager.set(_memory_resource.name, memory)
    await manager.set(_services_resource.name, services)
    return manager

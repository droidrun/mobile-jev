"""The two live dependencies resolved for a workflow invocation."""

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated
from uuid import uuid4

from llama_index.core.memory import Memory
from mobilerun.core.devices import Device
from workflows.resource import Resource

from mobilerun_agents.agent.memory.blocks.repeat import RepeatBlock
from mobilerun_agents.agent.memory.blocks.run_notes import RunNotesBlock
from mobilerun_agents.agent.tools.helpers.images import ImageStore

from .core.state import ActionRecord

active_device: ContextVar[Device] = ContextVar("agent_device")
active_memory_factory: ContextVar[Callable[[], Memory]] = ContextVar("agent_memory_factory")
active_images: ContextVar[ImageStore | None] = ContextVar("agent_images", default=None)
record_action: ContextVar[Callable[[ActionRecord], Awaitable[None]] | None] = ContextVar(
    "record_action", default=None
)
record_offset: ContextVar[int] = ContextVar("record_offset", default=0)
trace_image: ContextVar[Callable[[bytes], None] | None] = ContextVar("trace_image", default=None)


@contextmanager
def run_image_scope() -> Iterator[ImageStore | None]:
    """Give a root workflow an image store unless its operator already owns one."""
    if active_images.get() is not None:
        yield None
        return
    images = ImageStore(uuid4().hex, keep=False)
    token = active_images.set(images)
    try:
        yield images
    finally:
        active_images.reset(token)


def get_device() -> Device:
    return active_device.get()


def get_memory() -> Memory:
    memory = active_memory_factory.get()()
    memory.memory_blocks.extend([RunNotesBlock(), RepeatBlock()])
    return memory


DeviceResource = Annotated[Device, Resource(get_device)]
MemoryResource = Annotated[Memory, Resource(get_memory)]

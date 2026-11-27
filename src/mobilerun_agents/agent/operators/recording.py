"""Resolve an agent's recording policy without taking ownership of its phone."""

from typing import Any, Protocol

from mobilerun.core.devices import Device, RecordingDevice
from mobilerun.core.runs import recording_policy


class _Recording(Protocol):
    @property
    def enabled(self) -> bool: ...


class _Config(Protocol):
    @property
    def recording(self) -> _Recording: ...


def resolve_recording(
    device: Device, record: bool | None, config: _Config
) -> tuple[Device, RecordingDevice[Any] | None, bool]:
    """Apply operator, environment and YAML precedence, subject to hard vetoes."""
    if not device.recording_allowed() or not recording_policy()[0]:
        return device, None, False
    if record is None:
        record = config.recording.enabled
    if not record:
        return device, None, False
    if (existing := device.recording_device()) is not None:
        return device, existing, False
    recorder = RecordingDevice(device, capture="actions", kind="agent", by="agent")
    return recorder, recorder, True

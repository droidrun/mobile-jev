from __future__ import annotations

from datetime import UTC, datetime
from functools import cached_property
from typing import Any

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.core.devices import Device
from mobilerun.core.state.snapshot import DeviceSnapshot
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.context import ContextItem, JSONValue
from mobilerun.models.recording import Fingerprint
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)


class StepSnapshot(BaseModel):
    """Adapt a core observation to live workflow state and prompt content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ui: UIState = Field(repr=False)
    screenshot: ImageBlock | None = Field(default=None, exclude=True, repr=False)
    device_context: list[ContextItem] = Field(default_factory=list)
    _screenshot_attempted: bool = PrivateAttr(default=False)

    @classmethod
    async def observe(cls, device: Device, screenshot: bool = False) -> StepSnapshot:
        observed = await DeviceSnapshot.observe(device, screenshot=screenshot, redact=True)
        snapshot = cls(
            at=observed.at,
            ui=observed.ui,
            device_context=observed.context,
            screenshot=ImageBlock(image=observed.screenshot) if observed.screenshot else None,
        )
        snapshot._screenshot_attempted = screenshot
        return snapshot

    async def capture_screenshot(self, device: Device) -> None:
        if self.screenshot is not None or self._screenshot_attempted:
            return
        self._screenshot_attempted = True
        try:
            image = await device.screenshot()
        except DeviceOperationNotSupported:
            return
        if image:
            self.screenshot = ImageBlock(image=image)

    @cached_property
    def fingerprint(self) -> Fingerprint | None:
        fingerprint = DeviceSnapshot(self.at, self.ui).fingerprint
        if fingerprint.app or fingerprint.activity or fingerprint.elements:
            return fingerprint
        return None

    def compare(self, other: StepSnapshot) -> bool | None:
        """Return whether the live screen changed without changing the structural fingerprint."""
        if (self.ui.formatted_text, self.ui.focused_text) != (
            other.ui.formatted_text,
            other.ui.focused_text,
        ):
            return True
        before, after = self.fingerprint, other.fingerprint
        return before != after if before is not None and after is not None else None

    @field_serializer("ui")
    def _dump_ui(self, ui: UIState) -> JSONValue:
        return DeviceSnapshot(self.at, ui).to_json_data()["ui"]

    @field_validator("ui", mode="before")
    @classmethod
    def _load_ui(cls, value: UIState | dict[str, Any]) -> UIState:
        match value:
            case dict():
                return UIState(**value)
            case _:
                return value

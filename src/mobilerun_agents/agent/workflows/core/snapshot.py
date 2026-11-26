from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import cached_property
from hashlib import sha256
from typing import Any

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.core.devices import Device
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.models.context import ContextItem, JSONValue
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator

_UI_FIELDS = (
    "elements",
    "formatted_text",
    "focused_text",
    "phone_state",
    "screen_width",
    "screen_height",
    "use_normalized",
    "coordinate_scale_x",
    "coordinate_scale_y",
    "coordinate_contract_active",
    "model_screenshot_width",
    "model_screenshot_height",
)

_ELEMENT_FIELDS = (
    ("text", "label", "value"),
    ("contentDescription", "content_desc", "content_description", "description"),
    ("resourceId", "resource_id", "viewIdResourceName", "id"),
    ("className", "class_name", "class", "type"),
    ("bounds", "boundsInScreen", "bounds_in_screen"),
    ("checkedState", "isChecked", "checked"),
    ("isSelected", "selected"),
    ("focused", "isFocused"),
    ("enabled", "isEnabled"),
)


def _element_structure(element: dict[str, JSONValue]) -> list[JSONValue]:
    match element.get("details"):
        case dict() as details:
            properties = {**details, **element}
        case _:
            properties = element
    fields = [
        next((properties[k] for k in aliases if k in properties), None)
        for aliases in _ELEMENT_FIELDS
    ]
    children: list[JSONValue] = []
    match element.get("children"):
        case list() as nodes:
            for child in nodes:
                match child:
                    case dict():
                        children.append(_element_structure(child))
    fields.append(children)
    return fields


class StepSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ui: UIState = Field(repr=False)
    screenshot: ImageBlock | None = Field(default=None, repr=False)
    device_context: list[ContextItem] = Field(default_factory=list)
    _screenshot_attempted: bool = PrivateAttr(default=False)

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
    def fingerprint(self) -> tuple[str, str] | None:
        phone = self.ui.phone_state
        location = sha256(
            json.dumps([phone.get("packageName"), phone.get("activityName")]).encode()
        ).hexdigest()
        evidence: JSONValue
        if self.ui.elements:
            evidence = [_element_structure(element) for element in self.ui.elements]
        elif self.screenshot is not None:
            try:
                with self.screenshot.resolve_image() as image:
                    evidence = sha256(image.read()).hexdigest()
            except OSError:
                return None
        else:
            return None
        raw = json.dumps(
            [evidence, self.ui.focused_text, phone.get("focusedElement")], sort_keys=True
        )
        return location, sha256(raw.encode()).hexdigest()

    def compare(self, other: StepSnapshot) -> bool | None:
        """Return whether the screen changed, or None if the evidence is incomparable."""
        if bool(self.ui.elements) != bool(other.ui.elements):
            return None
        before, after = self.fingerprint, other.fingerprint
        return before != after if before is not None and after is not None else None

    @field_serializer("screenshot")
    def _dump_screenshot(self, screenshot: ImageBlock | None) -> dict[str, Any] | None:
        # A deferred in-memory screenshot can be used for comparison, but is
        # never included in a context payload, even by an early checkpoint.
        if screenshot is None or screenshot.path is None:
            return None
        return screenshot.model_dump(exclude={"image", "url"})

    @field_serializer("ui")
    def _dump_ui(self, ui: UIState) -> dict[str, Any]:
        return {name: getattr(ui, name) for name in _UI_FIELDS}

    @field_validator("ui", mode="before")
    @classmethod
    def _load_ui(cls, value: Any) -> Any:
        match value:
            case dict():
                return UIState(**{name: value[name] for name in _UI_FIELDS if name in value})
            case _:
                return value

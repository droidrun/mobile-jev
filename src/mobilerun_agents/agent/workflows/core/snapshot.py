from __future__ import annotations

from datetime import UTC, datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, field_validator

from .screen import screen_fingerprint

if TYPE_CHECKING:
    from .state import ActionRecord


class StepSnapshot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ui: UIState = Field(repr=False)
    screenshot: ImageBlock | None = Field(default=None, repr=False)
    notes: list[str] = Field(default_factory=list)
    goal: str
    step: int
    device_context: list[ContextItem]
    _actions: tuple[ActionRecord, ...] = PrivateAttr(default=())
    _thresholds: tuple[int, int, int] = PrivateAttr(default=(2, 3, 3))

    @cached_property
    def fingerprint(self) -> tuple[str, str]:
        return screen_fingerprint(self.ui, self.screenshot)

    def attach_screenshot(self, screenshot: ImageBlock | None) -> None:
        self.screenshot = screenshot
        if not self.ui.elements:
            self.__dict__.pop("fingerprint", None)

    @field_serializer("ui")
    def _dump_ui(self, ui: UIState) -> dict[str, Any]:
        return vars(ui).copy()

    @field_validator("ui", mode="before")
    @classmethod
    def _load_ui(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        fields = dict(value)
        focused = fields.pop("_has_single_focused_app_node", None)
        ui = UIState(**fields)
        if focused is not None:
            vars(ui)["_has_single_focused_app_node"] = focused
        return ui

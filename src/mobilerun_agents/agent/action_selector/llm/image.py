"""A model-local image and its inverse mapping to device input coordinates."""

from collections.abc import Collection
from dataclasses import dataclass

from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import JSONValue
from mobilerun.utils.images import (
    anthropic_resized_size,
    fit_dimensions_to_max_side,
    image_dimensions,
    resize_image_to_dimensions_with_grid,
)

from mobilerun_agents.agent.tools.helpers import coordinate_pairs, device_dimensions

from .models import Capabilities
from .parsing import ToolCall


@dataclass(frozen=True)
class ModelImage:
    data: bytes | None
    width: int
    height: int
    normalized: bool
    device: tuple[int, int]

    def point(self, x: JSONValue, y: JSONValue) -> tuple[int, int]:
        match x, y:
            case int(px), int(py) if type(x) is int and type(y) is int:
                pass
            case _:
                raise ValueError("Expected integer coordinates")
        w, h = (1000, 1000) if self.normalized else (self.width, self.height)
        if not (0 <= px < w + self.normalized and 0 <= py < h + self.normalized):
            raise ValueError("Coordinates outside the model image")
        dw, dh = self.device
        return min(dw - 1, round(px * dw / w)), min(dh - 1, round(py * dh / h))

    def to_device(self, call: ToolCall, fields: Collection[str]) -> None:
        args = call.parameters
        for x, y in coordinate_pairs(fields):
            if x in args and y in args and args[x] is not None and args[y] is not None:
                args[x], args[y] = self.point(args[x], args[y])


def prepare_image(
    ui: UIState, screenshot_only: bool, limits: Capabilities, data: bytes | None
) -> ModelImage:
    device = device_dimensions(ui, screenshot_only)
    if data is None:
        return ModelImage(
            None,
            ui.model_screenshot_width or device[0],
            ui.model_screenshot_height or device[1],
            ui.use_normalized,
            device,
        )
    width, height = image_dimensions(data)
    if limits.image_token_limit and limits.max_image_side:
        width, height = anthropic_resized_size(
            width, height, limits.max_image_side, limits.image_token_limit
        )
    for limit in (limits.max_image_side, limits.grounding_max_side):
        if limit:
            width, height = fit_dimensions_to_max_side(width, height, limit)
    data = resize_image_to_dimensions_with_grid(
        data, width, height, divisions=0 if limits.normalized_coords else 10
    )
    return ModelImage(data, width, height, limits.normalized_coords, device)

"""Validate device-input pixels against the pinned observation."""

from collections.abc import Collection

from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_prompt


def coordinate_pairs(fields: Collection[str]) -> tuple[tuple[str, str], ...]:
    if {"x1", "y1", "x2", "y2"} <= set(fields):
        return (("x1", "y1"), ("x2", "y2"))
    if {"x", "y"} <= set(fields):
        return (("x", "y"),)
    return ()


def device_dimensions(ui: UIState, screenshot_only: bool) -> tuple[int, int]:
    if screenshot_only and not ui.use_normalized:
        return round(ui.screen_width * ui.coordinate_scale_x), round(
            ui.screen_height * ui.coordinate_scale_y
        )
    return ui.screen_width, ui.screen_height


def check_point(x: int, y: int, ctx: Run) -> tuple[int, int]:
    provider = ctx.state_provider
    if provider.requires_active_contract_for_coords and not ctx.ui.coordinate_contract_active:
        raise ValueError(
            render_prompt("assets/prompts/tools/coordinate_error.jinja2", kind="inactive")
        )
    width, height = device_dimensions(ctx.ui, provider.requires_coordinate_tools)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            render_prompt(
                "assets/prompts/tools/coordinate_error.jinja2",
                kind="model_bounds",
                x=x,
                y=y,
                width=width,
                height=height,
            )
        )
    return x, y

"""Expand atomic choices for indexed taps, swipes and app launches."""

from collections.abc import Awaitable, Callable

from mobilerun.core.devices import Device
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceOperationNotSupported

from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig, ToolsConfig

from ..helpers import device_dimensions, element_targets
from .schema import AtomicTool


def _dimensions(device: Device, ui: UIState | None) -> tuple[int, int] | None:
    if ui is None or not ui.screen_width or not ui.screen_height:
        return None
    provider = device.state._resolved_provider
    if provider.requires_active_contract_for_coords and not ui.coordinate_contract_active:
        return None
    width, height = device_dimensions(ui, provider.requires_coordinate_tools)
    return (width, height) if min(width, height) >= 4 else None


async def _elements(
    device: Device, ui: UIState | None, goal: str, tools: ToolsConfig, decision: DecisionConfig
) -> list[AtomicTool]:
    if ui is None:
        return []
    return [
        AtomicTool("tap_element", {"index": index}, label)
        for index, label in element_targets(ui, decision)
    ]


async def _apps(
    device: Device, ui: UIState | None, goal: str, tools: ToolsConfig, decision: DecisionConfig
) -> list[AtomicTool]:
    goal = goal.casefold()
    try:
        records = await device.list_apps(include_system=True)
    except DeviceOperationNotSupported:
        return []
    apps = [
        (identifier, record.get("label") or identifier)
        for record in records
        if (identifier := record.get("package_name") or record.get("package"))
    ]
    return [
        AtomicTool(
            "start_app",
            {"package": identifier},
            render_prompt(
                decision.candidate_prompt,
                kind="app",
                label=label,
                identifier=identifier,
            ),
        )
        for identifier, label in sorted(
            apps,
            key=lambda app: not (len(app[1]) > 1 and app[1].casefold() in goal),
        )
    ]


async def _scrolls(
    device: Device, ui: UIState | None, goal: str, tools: ToolsConfig, decision: DecisionConfig
) -> list[AtomicTool]:
    if (space := _dimensions(device, ui)) is None:
        return []
    width, height = space
    return [
        AtomicTool(
            "swipe",
            {
                "x1": round((width - 1) * start[0]),
                "y1": round((height - 1) * start[1]),
                "x2": round((width - 1) * end[0]),
                "y2": round((height - 1) * end[1]),
                "duration_ms": tools.swipe_duration * 1000,
            },
            render_prompt(decision.candidate_prompt, kind="scroll", direction=direction),
        )
        for direction, (start, end) in tools.swipe_paths.items()
    ]


async def _completion(
    device: Device, ui: UIState | None, goal: str, tools: ToolsConfig, decision: DecisionConfig
) -> list[AtomicTool]:
    return [
        AtomicTool(
            "complete",
            {"success": success},
            render_prompt(decision.criteria_prompt, kind=kind),
        )
        for success, kind in ((True, "complete_success"), (False, "complete_failure"))
    ]


ATOMIC_EXCEPTIONS: dict[
    str,
    Callable[
        [Device, UIState | None, str, ToolsConfig, DecisionConfig], Awaitable[list[AtomicTool]]
    ],
] = {
    "complete": _completion,
    "tap_element": _elements,
    "start_app": _apps,
    "swipe": _scrolls,
}

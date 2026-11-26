"""Expand atomic choices for indexed taps, swipes and app launches."""

from collections.abc import Awaitable, Callable

from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_prompt

from ..helpers import device_dimensions, element_targets
from .schema import AtomicTool


def _dimensions(run: Run) -> tuple[int, int] | None:
    ui = run.ui
    if ui is None or not ui.screen_width or not ui.screen_height:
        return None
    provider = run.state_provider
    if provider.requires_active_contract_for_coords and not ui.coordinate_contract_active:
        return None
    width, height = device_dimensions(ui, provider.requires_coordinate_tools)
    return (width, height) if min(width, height) >= 4 else None


async def _elements(run: Run) -> list[AtomicTool]:
    if run.ui is None:
        return []
    return [
        AtomicTool("tap_element", {"index": index}, label)
        for index, label in element_targets(run.ui, run.config.decision)
    ]


async def _apps(run: Run) -> list[AtomicTool]:
    goal = run.state.goal.casefold()
    return [
        AtomicTool(
            "start_app",
            {"package": app.identifier},
            render_prompt(
                run.config.decision.candidate_prompt,
                kind="app",
                label=app.label,
                identifier=app.identifier,
            ),
        )
        for app in sorted(
            await run.installed_apps(),
            key=lambda a: not (len(a.label) > 1 and a.label.casefold() in goal),
        )
    ]


async def _scrolls(run: Run) -> list[AtomicTool]:
    if (space := _dimensions(run)) is None:
        return []
    width, height = space
    config = run.tool_config
    return [
        AtomicTool(
            "swipe",
            {
                "x1": round((width - 1) * start[0]),
                "y1": round((height - 1) * start[1]),
                "x2": round((width - 1) * end[0]),
                "y2": round((height - 1) * end[1]),
                "duration_ms": config.swipe_duration * 1000,
            },
            render_prompt(
                run.config.decision.candidate_prompt, kind="scroll", direction=direction
            ),
        )
        for direction, (start, end) in config.swipe_paths.items()
    ]


async def _completion(run: Run) -> list[AtomicTool]:
    return [
        AtomicTool(
            "complete",
            {"success": success},
            render_prompt(run.config.decision.criteria_prompt, kind=kind),
        )
        for success, kind in ((True, "complete_success"), (False, "complete_failure"))
    ]


ATOMIC_EXCEPTIONS: dict[str, Callable[[Run], Awaitable[list[AtomicTool]]]] = {
    "complete": _completion,
    "tap_element": _elements,
    "start_app": _apps,
    "swipe": _scrolls,
}

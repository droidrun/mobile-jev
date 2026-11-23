"""Action functions for device interaction.

Each function receives ``ctx: ActionContext`` as a keyword argument and
interacts with the device via ``ctx.driver``, resolves UI elements via
``ctx.ui``, and accesses shared state via ``ctx.shared_state``.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from mobilerun.core.state.ui_state import UIState
    from mobilerun.macro.recorder import MacroRecorder

    from mobilerun.agents.tools.action_context import ActionContext

from mobilerun.utils.coordinates import NORMALIZED_MAX, to_normalized

from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.tools.action_result import ActionResult
from mobilerun.agents.workflows.oneflows.app_starter_workflow import AppStarter

logger = logging.getLogger("mobilerun")


# ---------------------------------------------------------------------------
# Core UI actions
# ---------------------------------------------------------------------------


def _require_ui(ctx: "ActionContext") -> "UIState":
    if ctx.ui is None:
        raise ValueError("UI state is unavailable; refresh the device state before acting")
    return ctx.ui


def _uses_screenshot_only_coordinates(ctx: "ActionContext") -> bool:
    return ctx.state_provider.requires_coordinate_tools


def _screenshot_only_coordinate_error(ctx: "ActionContext") -> str:
    if ctx.ui is not None and ctx.ui.use_normalized:
        return (
            "Coordinates must be normalized values from 0 to 1000 on both axes "
            "in screenshot-only mode. Observe the screenshot and retry with "
            "normalized coordinates."
        )
    width = ctx.ui.screen_width if ctx.ui is not None else None
    height = ctx.ui.screen_height if ctx.ui is not None else None
    if width and height:
        return (
            f"Coordinates must be inside the screenshot size {width}x{height} "
            "in screenshot-only mode. Observe the screenshot and retry with "
            "pixel coordinates inside the image."
        )
    return (
        "Coordinates must be inside the screenshot bounds in screenshot-only mode. "
        "Observe the screenshot and retry with pixel coordinates inside the image."
    )


def _validate_screenshot_only_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> None:
    if not _uses_screenshot_only_coordinates(ctx):
        return
    try:
        ui = _require_ui(ctx)
        width = float(ui.screen_width)
        height = float(ui.screen_height)
        px = float(x)
        py = float(y)
    except TypeError as exc:
        raise ValueError(_screenshot_only_coordinate_error(ctx)) from exc
    except ValueError as exc:
        raise ValueError(_screenshot_only_coordinate_error(ctx)) from exc

    if ctx.ui is not None and ctx.ui.use_normalized:
        if not (0 <= px <= NORMALIZED_MAX and 0 <= py <= NORMALIZED_MAX):
            raise ValueError(_screenshot_only_coordinate_error(ctx))
        return

    out_of_range = width <= 0 or height <= 0 or px < 0 or px >= width or py < 0 or py >= height
    if out_of_range:
        raise ValueError(_screenshot_only_coordinate_error(ctx))


def _model_space_dimensions(ctx: "ActionContext") -> tuple[float, float] | None:
    """Return the (width, height) coordinate space the model was shown.

    Only applies when the provider declared a scaled coordinate contract
    (``resize_model_screenshot``) outside screenshot-only mode: the UIState
    keeps native screen dimensions while the model sees and answers in the
    ``native / coordinate_scale`` display space.
    """
    if _uses_screenshot_only_coordinates(ctx):
        return None
    # Read the contract state from the UIState snapshot the tap uses.
    if not (ctx.ui is not None and ctx.ui.coordinate_contract_active):
        return None
    if ctx.ui is not None and ctx.ui.use_normalized:
        return None
    try:
        scale_x = float(ctx.ui.coordinate_scale_x)
        scale_y = float(ctx.ui.coordinate_scale_y)
        width = float(ctx.ui.screen_width) / scale_x
        height = float(ctx.ui.screen_height) / scale_y
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _validate_model_space_point(x: int | float, y: int | float, *, ctx: "ActionContext") -> None:
    dims = _model_space_dimensions(ctx)
    if dims is None:
        return
    width, height = dims
    try:
        px = float(x)
        py = float(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Coordinates must be numbers inside the {round(width)}x{round(height)} "
            "coordinate space of the provided device state."
        ) from exc
    if px < 0 or px >= width or py < 0 or py >= height:
        raise ValueError(
            f"Coordinates ({x}, {y}) are outside the {round(width)}x{round(height)} "
            "coordinate space of the provided device state. Use the element "
            "bounds and screenshot shown to you and retry with coordinates "
            "inside that space."
        )


def _require_active_coordinate_contract(ctx: "ActionContext") -> None:
    """Reject coordinate actions when the provider requires the vision
    coordinate contract but it is not active for the current state.

    Some providers (e.g. iOS) can only map model coordinates to the tap input
    space while the contract is active. Without it, refuse the action so the
    model retries or uses an element-index ``click`` instead of tapping the
    wrong location."""
    provider = ctx.state_provider
    if not provider.requires_active_contract_for_coords:
        return
    # Read the contract state from the UIState snapshot the tap uses, not a
    # mutable provider flag.
    if ctx.ui is not None and ctx.ui.coordinate_contract_active:
        return
    raise ValueError(
        "Coordinate actions are unavailable for this step — the screen state "
        "could not be captured fully. Use an element index with `click`, or "
        "retry to get a fresh screen state."
    )


def _validate_normalized_point(x: int | float, y: int | float, *, ctx: "ActionContext") -> None:
    # Screenshot-only mode reports its own error.
    if _uses_screenshot_only_coordinates(ctx):
        return
    if not (ctx.ui is not None and ctx.ui.use_normalized):
        return
    try:
        inside = 0 <= float(x) <= NORMALIZED_MAX and 0 <= float(y) <= NORMALIZED_MAX
    except (TypeError, ValueError):
        inside = False
    if not inside:
        raise ValueError(
            f"Coordinates ({x}, {y}) are outside the normalized 0-1000 range "
            "of the device state. Use the element bounds shown to you and retry."
        )


def _convert_action_point(
    x: int | float, y: int | float, *, ctx: "ActionContext"
) -> tuple[int, int]:
    _validate_screenshot_only_point(x, y, ctx=ctx)
    _validate_normalized_point(x, y, ctx=ctx)
    _require_active_coordinate_contract(ctx)
    _validate_model_space_point(x, y, ctx=ctx)
    abs_x, abs_y = _require_ui(ctx).convert_point(x, y)
    return int(round(abs_x)), int(round(abs_y))


def _summary_point(
    x: int | float, y: int | float, abs_x: int, abs_y: int, *, ctx: "ActionContext"
) -> str:
    # Normalized callers get their own 0-1000 values back.
    if ctx.ui is not None and ctx.ui.use_normalized:
        return f"({x}, {y})"
    return f"({abs_x}, {abs_y})"


def _element_summary_point(x: int, y: int, *, ctx: "ActionContext") -> str:
    # Element taps are device units; normalized callers see 0-1000.
    ui = ctx.ui
    if (ui is not None and ui.use_normalized) and ui.screen_width and ui.screen_height:
        x, y = to_normalized(x, y, ui.screen_width, ui.screen_height)
    return f"({x}, {y})"


def _macro_recorder(ctx: "ActionContext") -> "MacroRecorder | None":
    return ctx.macro_recorder


def _record_macro_action(
    ctx: "ActionContext",
    action: dict[str, Any],
    *,
    pre_ui: "UIState | None" = None,
) -> None:
    recorder = _macro_recorder(ctx)
    if recorder is None:
        return
    recorder.record_action(
        action,
        pre_ui=pre_ui if pre_ui is not None else ctx.ui,
    )


async def _macro_pre_ui(ctx: "ActionContext") -> "UIState | None":
    if _macro_recorder(ctx) is None:
        return ctx.ui
    state_provider = ctx.state_provider
    try:
        return await state_provider.get_state()
    except Exception as e:
        logger.debug(f"Failed to refresh macro pre-state: {e}")
        return ctx.ui


async def _macro_pre_ui_after_focus_tap(ctx: "ActionContext") -> "UIState | None":
    if _macro_recorder(ctx) is not None:
        await asyncio.sleep(get_agent_settings().MOBILERUN_MACRO_FOCUS_SETTLE_SECONDS)
    return await _macro_pre_ui(ctx)


def _driver_log_length(ctx: "ActionContext") -> int | None:
    recording = ctx.driver._recording
    return len(recording.recorded_actions) if recording is not None else None


def _record_driver_log_delta(
    ctx: "ActionContext", before: int | None, *, pre_ui: "UIState | None" = None
) -> None:
    if before is None:
        return
    recording = ctx.driver._recording
    if recording is None:
        return
    for raw_action in recording.recorded_actions[before:]:
        _record_macro_action(ctx, dict(raw_action), pre_ui=pre_ui)


async def click(index: int, *, ctx: "ActionContext") -> ActionResult:
    """Click the element with the given index."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        ui = _require_ui(ctx)
        x, y = ui.get_element_coords(index)
        await ctx.driver.actions.call("device.tap", {"x": x, "y": y}, audience="operator")
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": x, "y": y},
            pre_ui=pre_ui,
        )

        info = ui.get_element_info(index)
        detail_parts = [
            f"Text: '{info.get('text', 'No text')}'",
            f"Class: {info.get('className', 'Unknown class')}",
            f"Type: {info.get('type', 'unknown')}",
        ]
        if info.get("child_texts"):
            detail_parts.append(f"Contains text: {' | '.join(info['child_texts'])}")
        detail_parts.append(f"Coordinates: {_element_summary_point(x, y, ctx=ctx)}")

        return ActionResult(success=True, summary=f"Clicked on {' | '.join(detail_parts)}")
    except ValueError as e:
        return ActionResult(success=False, summary=f"Failed to click element at index {index}: {e}")


async def long_press(index: int, *, ctx: "ActionContext") -> ActionResult:
    """Long press the element with the given index."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        x, y = _require_ui(ctx).get_element_coords(index)
        await ctx.driver.actions.call(
            "device.swipe",
            {"x1": x, "y1": y, "x2": x, "y2": y, "duration_ms": 1000},
            audience="operator",
        )
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": x,
                "start_y": y,
                "end_x": x,
                "end_y": y,
                "duration_ms": 1000,
            },
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True,
            summary=(
                f"Long pressed element at index {index} at {_element_summary_point(x, y, ctx=ctx)}"
            ),
        )
    except ValueError as e:
        return ActionResult(
            success=False, summary=f"Failed to long press element at index {index}: {e}"
        )


async def long_press_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Long press at screen coordinates."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.actions.call(
            "device.swipe",
            {"x1": abs_x, "y1": abs_y, "x2": abs_x, "y2": abs_y, "duration_ms": 1000},
            audience="operator",
        )
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": abs_x,
                "start_y": abs_y,
                "end_x": abs_x,
                "end_y": abs_y,
                "duration_ms": 1000,
            },
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True,
            summary=f"Long pressed at {_summary_point(x, y, abs_x, abs_y, ctx=ctx)}",
        )
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to long press at ({x}, {y}): {describe_error(e)}",
        )


async def click_at(x: int, y: int, *, ctx: "ActionContext") -> ActionResult:
    """Click at screen coordinates."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        abs_x, abs_y = _convert_action_point(x, y, ctx=ctx)
        await ctx.driver.actions.call("device.tap", {"x": abs_x, "y": abs_y}, audience="operator")
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": abs_x, "y": abs_y},
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True,
            summary=f"Tapped at {_summary_point(x, y, abs_x, abs_y, ctx=ctx)}",
        )
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to tap at ({x}, {y}): {describe_error(e)}"
        )


async def click_area(x1: int, y1: int, x2: int, y2: int, *, ctx: "ActionContext") -> ActionResult:
    """Click center of area."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        _validate_screenshot_only_point(x1, y1, ctx=ctx)
        _validate_screenshot_only_point(x2, y2, ctx=ctx)
        _validate_normalized_point(x1, y1, ctx=ctx)
        _validate_normalized_point(x2, y2, ctx=ctx)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        abs_x, abs_y = _convert_action_point(cx, cy, ctx=ctx)
        await ctx.driver.actions.call("device.tap", {"x": abs_x, "y": abs_y}, audience="operator")
        _record_macro_action(
            ctx,
            {"action_type": "tap", "x": abs_x, "y": abs_y},
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True,
            summary=(f"Tapped center of area at {_summary_point(cx, cy, abs_x, abs_y, ctx=ctx)}"),
        )
    except Exception as e:
        return ActionResult(
            success=False, summary=f"Failed to tap area center: {describe_error(e)}"
        )


async def type_text(
    text: str, index: int | None = None, clear: bool = False, *, ctx: "ActionContext"
) -> ActionResult:
    """Type text into an indexed element or the currently focused input."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        if index is not None and index != -1:
            x, y = _require_ui(ctx).get_element_coords(index)
            await ctx.driver.actions.call("device.tap", {"x": x, "y": y}, audience="operator")
            _record_macro_action(
                ctx,
                {"action_type": "tap", "x": x, "y": y},
                pre_ui=pre_ui,
            )
            pre_ui = await _macro_pre_ui_after_focus_tap(ctx)

        success = await ctx.driver.actions.call(
            "device.input_text", {"text": text, "clear": clear}, audience="operator"
        )
        if success:
            _record_macro_action(
                ctx,
                {"action_type": "input_text", "text": text, "clear": clear},
                pre_ui=pre_ui,
            )
            return ActionResult(success=True, summary=f"Text typed successfully (clear={clear})")
        else:
            return ActionResult(success=False, summary="Failed to type text: input failed")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to type text: {describe_error(e)}")


async def type_text_direct(text: str, clear: bool = False, *, ctx: "ActionContext") -> ActionResult:
    """Type text into the currently focused input."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        success = await ctx.driver.actions.call(
            "device.input_text", {"text": text, "clear": clear}, audience="operator"
        )
        if success:
            _record_macro_action(
                ctx,
                {"action_type": "input_text", "text": text, "clear": clear},
                pre_ui=pre_ui,
            )
            return ActionResult(success=True, summary=f"Text typed successfully (clear={clear})")
        return ActionResult(success=False, summary="Failed to type text: input failed")
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to type text: {describe_error(e)}")


async def system_button(button: str, *, ctx: "ActionContext") -> ActionResult:
    """Press a system button (back, home, or enter)."""
    try:
        pre_ui = await _macro_pre_ui(ctx)
        await ctx.driver.actions.call(
            "device.press_button", {"button": button}, audience="operator"
        )
        _record_macro_action(
            ctx,
            {"action_type": "button_press", "button": button},
            pre_ui=pre_ui,
        )
        return ActionResult(success=True, summary=f"Pressed {button.upper()} button")
    except ValueError as e:
        return ActionResult(success=False, summary=str(e))
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to press {button} button: {e.__class__.__name__}: {e}",
        )


async def swipe(
    coordinate: List[int],
    coordinate2: List[int],
    duration: float = 1.0,
    *,
    ctx: "ActionContext",
) -> ActionResult:
    """Swipe from one coordinate to another."""
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        return ActionResult(
            success=False,
            summary=f"Failed: coordinate must be a list of 2 integers, got: {coordinate}",
        )
    if not isinstance(coordinate2, list) or len(coordinate2) != 2:
        return ActionResult(
            success=False,
            summary=f"Failed: coordinate2 must be a list of 2 integers, got: {coordinate2}",
        )

    try:
        pre_ui = await _macro_pre_ui(ctx)
        start_x, start_y = _convert_action_point(*coordinate, ctx=ctx)
        end_x, end_y = _convert_action_point(*coordinate2, ctx=ctx)
        duration_ms = int(duration * 1000)
        await ctx.driver.actions.call(
            "device.swipe",
            {
                "x1": start_x,
                "y1": start_y,
                "x2": end_x,
                "y2": end_y,
                "duration_ms": duration_ms,
            },
            audience="operator",
        )
        _record_macro_action(
            ctx,
            {
                "action_type": "swipe",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "duration_ms": duration_ms,
            },
            pre_ui=pre_ui,
        )
        return ActionResult(
            success=True,
            summary=(
                "Swiped from "
                f"{_summary_point(coordinate[0], coordinate[1], start_x, start_y, ctx=ctx)} to "
                f"{_summary_point(coordinate2[0], coordinate2[1], end_x, end_y, ctx=ctx)}"
            ),
        )
    except Exception as e:
        return ActionResult(success=False, summary=f"Failed to swipe: {describe_error(e)}")


async def open_app(text: str, *, ctx: "ActionContext") -> ActionResult:
    """Open an app by its name."""
    if ctx.app_opener_llm is None:
        return ActionResult(
            success=False,
            summary="Failed: app_opener_llm not configured.",
        )

    workflow = AppStarter(
        driver=ctx.driver,
        llm=ctx.app_opener_llm,
        timeout=60,
        stream=ctx.streaming,
        verbose=False,
    )

    pre_ui = await _macro_pre_ui(ctx)
    driver_log_before = _driver_log_length(ctx)
    result = await workflow.run(app_description=text)
    await asyncio.sleep(1)

    if isinstance(result, str) and "could not open app" in result.lower():
        return ActionResult(success=False, summary=result)
    _record_driver_log_delta(ctx, driver_log_before, pre_ui=pre_ui)
    return ActionResult(success=True, summary=str(result))


async def open_bundle_id(
    bundle_id: str | None = None,
    app_id: str | None = None,
    *,
    ctx: "ActionContext",
) -> ActionResult:
    """Open an app by exact package name, app id, or iOS bundle identifier."""
    # Models sometimes end parameter values with a newline.
    identifier = str(app_id or bundle_id or "").strip()
    if not identifier:
        return ActionResult(
            success=False,
            summary="Failed to open app: exact app identifier is required.",
        )

    hint = (
        "Maybe you got the wrong app identifier. You could try using swipes and "
        "search to find the app."
    )
    try:
        pre_ui = await _macro_pre_ui(ctx)
        result = await ctx.driver.actions.call(
            "device.start_app", {"package": identifier}, audience="operator"
        )
        await asyncio.sleep(1)
        if isinstance(result, str) and result.lower().startswith("failed"):
            return ActionResult(
                success=False,
                summary=f"Failed to open app '{identifier}': {result}\n{hint}",
            )
        _record_macro_action(
            ctx,
            {"action_type": "start_app", "package": identifier, "activity": None},
            pre_ui=pre_ui,
        )
        return ActionResult(success=True, summary=str(result))
    except Exception as e:
        return ActionResult(
            success=False,
            summary=f"Failed to open app '{identifier}': {e.__class__.__name__}: {e}\n{hint}",
        )


async def wait(duration: float = 1.0, *, ctx: "ActionContext") -> ActionResult:
    """Wait for a specified duration in seconds."""
    pre_ui = await _macro_pre_ui(ctx)
    await ctx.driver.actions.call("device.wait", {"seconds": duration}, audience="operator")
    recorder = _macro_recorder(ctx)
    if recorder is not None:
        recorder.record_wait(duration, pre_ui=pre_ui)
    return ActionResult(success=True, summary=f"Waited for {duration} seconds")


# ---------------------------------------------------------------------------
# State / memory actions
# ---------------------------------------------------------------------------


async def complete(
    success: bool, reason: str = "", message: str = "", *, ctx: "ActionContext"
) -> ActionResult:
    """Mark the task as complete.

    ``message`` is the final answer/finding; ``reason`` is status or failure detail.
    """
    await ctx.shared_state.complete(success, reason=reason, message=message)
    return ActionResult(success=True, summary=ctx.shared_state.completion_reason)

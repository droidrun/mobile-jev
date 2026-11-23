"""Tests for the a11y+vision coordinate contract (issue #350).

When vision is enabled on AndroidStateProvider, the screenshot the agents
attach is resized (with a labeled grid) into a deterministic display space;
the provider declares that space in formatted_text, emits model-facing
bounds in it, and convert_point scales the model's coordinates back to
native device pixels. Element-index taps keep using native bounds.
"""

import asyncio
import inspect
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.base import StateProvider, _should_resize_model_screenshot
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.utils.images import (
    image_dimensions,
    resize_image_to_max_side_with_grid,
)
from PIL import Image

from mobilerun.agents.tools import actions as actions_module
from mobilerun.agents.tools.actions import _convert_action_point

NATIVE_W, NATIVE_H = 1080, 2400


def _combined_data(width=NATIVE_W, height=NATIVE_H):
    return {
        "a11y_tree": {
            "boundsInScreen": {"left": 0, "top": 0, "right": width, "bottom": height},
            "className": "android.widget.FrameLayout",
            "text": "root",
            "children": [
                {
                    "boundsInScreen": {
                        "left": 221,
                        "top": 409,
                        "right": 600,
                        "bottom": 460,
                    },
                    "className": "android.widget.TextView",
                    "text": "Services & preferences",
                    "children": [],
                }
            ],
        },
        "device_context": {"screen_bounds": {"width": width, "height": height}},
        "phone_state": {
            "currentApp": "Settings",
            "packageName": "com.android.settings",
        },
    }


def _provider(width=NATIVE_W, height=NATIVE_H, **kwargs):
    driver = SimpleNamespace(
        _state_recovery=None,
        _get_ui_tree=AsyncMock(return_value=_combined_data(width, height)),
    )
    return AndroidStateProvider(
        driver,
        tree_filter=SimpleNamespace(filter=lambda tree, ctx: tree),
        tree_formatter=IndexedFormatter(),
        **kwargs,
    )


def _get_state(provider):
    return asyncio.run(provider.get_state())


def test_resize_flags_pair_providers_with_agent_gating():
    # The providers declare the flag…
    assert StateProvider.resize_model_screenshot is False
    assert ScreenshotOnlyStateProvider.resize_model_screenshot is True
    assert _provider(vision_enabled=True).resize_model_screenshot is True
    assert _provider().resize_model_screenshot is False

    # …and every agent that attaches screenshots gates its resize on the
    # shared helper. (This pairing is what makes coordinate_scale valid.)
    import mobilerun.agents.workflows.executor.executor_agent as executor_agent
    import mobilerun.agents.workflows.fast_agent.fast_agent as fast_agent
    import mobilerun.agents.workflows.manager.manager_agent as manager_agent
    import mobilerun.agents.workflows.manager.stateless_manager_agent as stateless_manager_agent

    for module in (fast_agent, executor_agent, manager_agent, stateless_manager_agent):
        source = inspect.getsource(module)
        assert "_resize_model_screenshot_async" in source, module.__name__


def test_hybrid_vision_zero_bounds_refuses_click_at():
    from mobilerun.agents.tools.actions import click_at

    provider = _provider(width=0, height=0, vision_enabled=True)
    provider._device.tap = AsyncMock()
    state = _get_state(provider)
    ctx = SimpleNamespace(
        ui=state,
        state_provider=provider,
        driver=provider._device,
        macro_recorder=None,
    )

    assert state.coordinate_contract_active is False
    result = asyncio.run(click_at(10, 20, ctx=ctx))

    assert result.success is False
    assert "unavailable for this step" in result.summary
    provider._device.tap.assert_not_awaited()


def test_legacy_injected_screenshot_providers_keep_getting_resized():
    """Injected providers that only implement the pre-existing
    ``requires_coordinate_tools`` contract must still get resized screenshots:
    the tool registry, prompts, and validation all still treat them as
    screenshot-coordinate mode."""

    class LegacyScreenshotProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(LegacyScreenshotProvider(SimpleNamespace()))

    class DuckTypedLegacyProvider(StateProvider):
        requires_coordinate_tools = True

    assert _should_resize_model_screenshot(DuckTypedLegacyProvider(SimpleNamespace()))

    # Non-coordinate providers are unaffected
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))
    assert not _should_resize_model_screenshot(_provider())
    assert not _should_resize_model_screenshot(StateProvider(SimpleNamespace()))

    # And the new contract still activates it
    assert _should_resize_model_screenshot(_provider(vision_enabled=True))
    assert _should_resize_model_screenshot(ScreenshotOnlyStateProvider(SimpleNamespace()))


def test_provider_scale_matches_what_agents_actually_send():
    """The invariant that PR #355 lacked: the provider's scale must describe
    the image the agents really attach."""
    buf = BytesIO()
    Image.new("RGB", (NATIVE_W, NATIVE_H)).save(buf, format="PNG")
    from mobilerun.config.settings import get_settings

    sent = resize_image_to_max_side_with_grid(
        buf.getvalue(), get_settings().MOBILERUN_MODEL_SCREENSHOT_MAX_SIDE
    )
    sent_w, sent_h = image_dimensions(sent)

    state = _get_state(_provider(vision_enabled=True))
    assert sent_w == round(NATIVE_W / state.coordinate_scale_x)
    assert sent_h == round(NATIVE_H / state.coordinate_scale_y)


def _action_ctx(state, provider):
    return SimpleNamespace(ui=state, state_provider=provider)


def test_convert_action_point_validates_model_space():
    provider = _provider(vision_enabled=True)
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    # In-space coordinates convert to native pixels
    assert _convert_action_point(461, 1024, ctx=ctx) == (540, 1200)

    # Out-of-space coordinates (native-space habits) are rejected with a
    # message instead of tapping off-screen
    try:
        _convert_action_point(1000, 2300, ctx=ctx)
    except ValueError as e:
        assert "922x2048" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-space point")


def test_convert_action_point_unchanged_without_contract():
    provider = _provider()
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    assert _convert_action_point(1000, 2300, ctx=ctx) == (1000, 2300)


def test_validation_helper_is_wired_into_convert():
    source = inspect.getsource(actions_module._convert_action_point)
    assert "_validate_model_space_point" in source

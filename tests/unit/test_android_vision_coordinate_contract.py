"""Tests for the a11y+vision coordinate contract (issue #350).

When vision is enabled on AndroidStateProvider, the screenshot the agents
attach is resized (with a labeled grid) into a deterministic display space;
the provider declares that space in formatted_text, emits model-facing
bounds in it, and convert_point scales the model's coordinates back to
native device pixels. Element-index taps keep using native bounds.
"""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.utils.images import (
    image_dimensions,
    resize_image_to_max_side_with_grid,
)
from PIL import Image

from mobilerun_agents.agent.tools.helpers import check_point
from tests.unit.device_fixtures import tool_run

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


def test_hybrid_vision_zero_bounds_refuses_click_at():
    from mobilerun_agents.agent.tools.helpers import check_point

    provider = _provider(width=0, height=0, vision_enabled=True)
    provider._device.tap = AsyncMock()
    state = _get_state(provider)
    ctx = tool_run(
        ui=state,
        state_provider=provider,
        driver=provider._device,
        macro_recorder=None,
    )

    assert state.coordinate_contract_active is False
    with pytest.raises(ValueError, match="unavailable for this step"):
        check_point(10, 20, ctx)
    provider._device.tap.assert_not_awaited()


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


def test_check_point_validates_device_space():
    provider = _provider(vision_enabled=True)
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    # In-space coordinates convert to native pixels
    assert check_point(540, 1200, ctx=ctx) == (540, 1200)

    # Out-of-space coordinates (native-space habits) are rejected with a
    # message instead of tapping off-screen
    try:
        check_point(1080, 2400, ctx=ctx)
    except ValueError as e:
        assert "1080x2400" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-space point")


def test_check_point_unchanged_without_contract():
    provider = _provider()
    state = _get_state(provider)
    ctx = _action_ctx(state, provider)

    assert check_point(1000, 2300, ctx=ctx) == (1000, 2300)

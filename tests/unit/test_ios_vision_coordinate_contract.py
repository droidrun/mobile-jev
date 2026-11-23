"""Tests for the iOS a11y+vision coordinate contract.

iOS screenshots are physical pixels while taps and a11y bounds are points,
so the display space is derived from the actual screenshot bytes and
``convert_point`` maps the model's display-space output back to POINTS.
Mirrors tests/unit/test_android_vision_coordinate_contract.py.
"""

import asyncio
from io import BytesIO
from types import SimpleNamespace

from mobilerun.core.state.base import _should_resize_model_screenshot
from mobilerun.core.state.ios import IOSStateProvider
from PIL import Image

from mobilerun.agents.tools.actions import _convert_action_point

POINTS_W, POINTS_H = 440, 956
PIXELS_W, PIXELS_H = 1320, 2868  # 3x physical screenshot

RAW_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Cell, {{20.0, 200.0}, {400.0, 44.0}}, label: 'General'
      StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
"""


def _png(width, height):
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeIOSDriver:
    def __init__(self, screenshot_bytes=None, screenshot_error=None, screenshot_plan=None):
        # screenshot_plan: per-call behaviors (bytes or Exception); the last
        # entry repeats. Overrides screenshot_bytes/screenshot_error.
        self._plan = screenshot_plan or (
            [screenshot_error] if screenshot_error else [screenshot_bytes]
        )
        self.screenshot_calls = 0
        self.ui_tree_error = None

    async def _get_ui_tree(self):
        if self.ui_tree_error:
            raise self.ui_tree_error
        return {
            "phone_state": {
                "currentApp": "Settings",
                "packageName": "com.apple.Preferences",
            },
            "device_context": {"screen_bounds": {"width": POINTS_W, "height": POINTS_H}},
            "a11y_tree": RAW_A11Y,
        }

    async def screenshot(self):
        step = self._plan[min(self.screenshot_calls, len(self._plan) - 1)]
        self.screenshot_calls += 1
        if isinstance(step, Exception):
            raise step
        return step


def _state(provider):
    return asyncio.run(provider.get_state())


def _general_element(state):
    return next(e for e in state.elements if e.get("text") == "General")


def test_resize_flag_pairs_with_agent_gating():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    assert _should_resize_model_screenshot(IOSStateProvider(driver, vision_enabled=True))
    assert not _should_resize_model_screenshot(IOSStateProvider(driver))
    assert not _should_resize_model_screenshot(
        IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    )


def test_convert_action_point_validates_ios_model_space():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # display-space coordinates convert to points
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)

    # raw physical-pixel coordinates (the pre-contract failure mode) are
    # outside the 943x2048 display space and get rejected
    try:
        _convert_action_point(1200, 2500, ctx=ctx)
    except ValueError as e:
        assert "943x2048" in str(e)
    else:
        raise AssertionError("expected ValueError for image-space point")


def test_click_at_auto_unmasks_with_vision_contract():
    from mobilerun.agents.config.user_config import (
        DEFAULT_DISABLED_TOOLS,
    )
    from mobilerun.agents.workflows.droid.droid_agent import _effective_disabled_tools

    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))

    # vision + contract -> click_at auto-enabled
    provider = IOSStateProvider(driver, vision_enabled=True)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=True, explicit=False
    )
    assert "click_at" not in effective
    assert "click_area" in effective  # only click_at auto-unmasks

    # no vision -> stays masked
    provider = IOSStateProvider(driver)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=False, explicit=False
    )
    assert "click_at" in effective

    # normalized mode + vision -> 0-1000 maps to points, click_at auto-enabled
    provider = IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    effective = _effective_disabled_tools(
        list(DEFAULT_DISABLED_TOOLS), provider, vision_enabled=True, explicit=False
    )
    assert "click_at" not in effective

    # explicit user list is honored verbatim
    provider = IOSStateProvider(driver, vision_enabled=True)
    effective = _effective_disabled_tools(
        ["click_at"], provider, vision_enabled=True, explicit=True
    )
    assert "click_at" in effective


def test_coordinate_actions_refused_when_contract_drops_for_a_step():
    """Registry exposes click_at under vision, but a step whose contract
    dropped (probe failure) must refuse coordinate actions rather than tap
    raw pixels at 1:1 into point space."""
    driver = FakeIOSDriver(
        screenshot_plan=[
            RuntimeError("probe failed this step"),  # contract drops
            _png(PIXELS_W, PIXELS_H),  # recovers next step
        ]
    )
    provider = IOSStateProvider(driver, vision_enabled=True)

    state = _state(provider)  # fallback state, contract inactive
    assert provider.resize_model_screenshot is False
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # An in-points coordinate that would otherwise tap fine is still refused,
    # because this step cannot guarantee the model's coordinate space.
    import pytest

    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(100, 200, ctx=ctx)

    # Next step recovers the contract -> coordinate actions work again.
    state2 = _state(provider)
    assert provider.resize_model_screenshot is True
    ctx2 = SimpleNamespace(ui=state2, state_provider=provider)
    assert _convert_action_point(471, 1024, ctx=ctx2) == (220, 478)


def test_active_contract_state_allows_coordinate_actions():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # contract active -> converts, no refusal
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)


def test_non_vision_ios_does_not_require_contract():
    # Without vision the provider never intends the contract, so the guard is
    # inert and (masked-by-default) coordinate paths behave as before.
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    assert provider.requires_active_contract_for_coords is False
    assert _convert_action_point(100, 200, ctx=ctx) == (100, 200)


def test_guard_reads_snapshot_not_mutable_provider_flag():
    """Regression: a mid-action get_state re-probe (macro pre-state) can flip
    provider.resize_model_screenshot without updating ctx.ui. The guard and
    model-space validator must follow ctx.ui (the snapshot the tap uses), so a
    provider flip cannot cause a wrong tap or a spurious refusal."""
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    good = _state(provider)  # contract active snapshot
    assert good.coordinate_contract_active is True
    ctx = SimpleNamespace(ui=good, state_provider=provider)

    # provider flag flipped False by a hypothetical macro re-probe -> the tap
    # still uses `good`, so it must NOT be spuriously refused.
    provider.resize_model_screenshot = False
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)

    # Inverse: a fallback snapshot must be refused even if the provider flag
    # currently reads True (re-probe succeeded after the step's snapshot).
    drop_driver = FakeIOSDriver(screenshot_error=RuntimeError("probe failed"))
    drop_provider = IOSStateProvider(drop_driver, vision_enabled=True)
    fallback = _state(drop_provider)
    assert fallback.coordinate_contract_active is False
    drop_provider.resize_model_screenshot = True  # mimic later re-probe success
    import pytest

    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(
            100, 200, ctx=SimpleNamespace(ui=fallback, state_provider=drop_provider)
        )


def test_swipe_refused_on_dropped_contract_step():
    """swipe (default-enabled) routes through the same guard."""

    from mobilerun.agents.tools.actions import swipe

    driver = FakeIOSDriver(screenshot_error=RuntimeError("probe failed"))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider, driver=driver, macro_recorder=None)

    result = asyncio.run(swipe([100, 200], [100, 600], ctx=ctx))
    assert result.success is False
    assert "unavailable for this step" in result.summary


def test_click_at_end_to_end_refused_then_works(monkeypatch):
    """End-to-end through the async click_at wrapper: refused on a dropped
    contract, taps on an active one. Taps are stubbed to avoid a real driver."""
    taps = []

    class _Driver(FakeIOSDriver):
        @property
        def actions(self):
            async def call(name, args, *, audience):
                assert name == "device.tap" and audience == "operator"
                return await self.tap(args["x"], args["y"])

            return SimpleNamespace(call=call)

        async def tap(self, x, y):
            taps.append((x, y))

    # active contract -> taps in points
    driver = _Driver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    from mobilerun.agents.tools.actions import click_at

    ctx = SimpleNamespace(ui=state, state_provider=provider, driver=driver, macro_recorder=None)
    ok = asyncio.run(click_at(471, 1024, ctx=ctx))
    assert ok.success and taps == [(220, 478)]

    # dropped contract -> refused, no tap
    taps.clear()
    drop = _Driver(screenshot_error=RuntimeError("probe failed"))
    drop_provider = IOSStateProvider(drop, vision_enabled=True)
    drop_state = _state(drop_provider)
    ctx2 = SimpleNamespace(
        ui=drop_state, state_provider=drop_provider, driver=drop, macro_recorder=None
    )
    refused = asyncio.run(click_at(100, 200, ctx=ctx2))
    assert refused.success is False and taps == []


def test_empty_state_from_ui_tree_failure_refuses_coordinate_actions():
    """ui_tree fetch failure returns the empty state (contract inactive);
    coordinate actions must be refused there too."""
    import pytest

    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    driver.ui_tree_error = RuntimeError("tree fetch failed")
    state = _state(provider)

    assert state.coordinate_contract_active is False
    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(10, 20, ctx=SimpleNamespace(ui=state, state_provider=provider))


def test_normalized_mode_shows_normalized_bounds_and_taps_points():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver, use_normalized=True)
    state = _state(provider)

    element = _general_element(state)
    assert element["bounds"] == "20,200,420,244"
    assert element["displayBounds"] == "45,209,954,255"
    assert "(normalized [0-1000])" in state.formatted_text
    assert "(45,209,954,255)" in state.formatted_text
    assert "(20,200,420,244)" not in state.formatted_text
    # element taps keep points
    assert state.get_element_coords(element["index"]) == (220, 222)
    # the displayed center fed back through a coordinate action lands inside
    ctx = SimpleNamespace(ui=state, state_provider=provider)
    x, y = _convert_action_point(499, 232, ctx=ctx)
    assert 20 <= x < 420 and 200 <= y < 244
    assert driver.screenshot_calls == 0


def test_normalized_mode_refuses_coordinates_on_a_failed_tree_read():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    provider = IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    ok = _state(provider)
    driver.ui_tree_error = RuntimeError("portal timeout")
    failed = _state(provider)

    ctx = SimpleNamespace(ui=ok, state_provider=provider)
    assert _convert_action_point(500, 500, ctx=ctx) == (220, 478)
    import pytest

    with pytest.raises(ValueError, match="unavailable for this step"):
        _convert_action_point(500, 500, ctx=SimpleNamespace(ui=failed, state_provider=provider))

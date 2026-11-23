"""Regression coverage through raw Portal data, formatting, and real actions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.filters import ConciseFilter
from mobilerun.core.state.formatters import IndexedFormatter

from mobilerun.agents.tools.actions import (
    click,
    long_press,
    type_text,
)


def node(text, bounds=(0, 0, 1000, 400), children=(), **properties):
    return {
        "text": text,
        "className": "android.widget.Button",
        "boundsInScreen": dict(zip(("left", "top", "right", "bottom"), bounds, strict=True)),
        "windowId": 7,
        "drawingOrder": 1,
        "isClickable": True,
        "isVisibleToUser": True,
        "children": list(children),
        **properties,
    }


def context(children, screen=(1000, 2000), **provider_kwargs):
    width, height = screen
    raw = {
        "a11y_tree": node("root", children=children, isClickable=False),
        "phone_state": {},
        "device_context": {"screen_bounds": {"width": width, "height": height}},
    }
    driver = SimpleNamespace(
        _state_recovery=None,
        _get_ui_tree=AsyncMock(return_value=raw),
        tap=AsyncMock(),
        swipe=AsyncMock(),
        input_text=AsyncMock(return_value=True),
    )

    async def call(name, arguments, *, audience):
        assert audience == "operator"
        if name == "device.tap":
            return await driver.tap(arguments["x"], arguments["y"])
        if name == "device.swipe":
            return await driver.swipe(
                arguments["x1"],
                arguments["y1"],
                arguments["x2"],
                arguments["y2"],
                arguments["duration_ms"],
            )
        if name == "device.input_text":
            return await driver.input_text(arguments["text"], arguments["clear"])
        raise AssertionError(name)

    driver.actions = SimpleNamespace(call=call)
    provider = AndroidStateProvider(driver, ConciseFilter(), IndexedFormatter(), **provider_kwargs)
    ui = asyncio.run(provider.get_state())
    return SimpleNamespace(ui=ui, driver=driver, state_provider=provider, macro_recorder=None)


def element(ctx, text):
    return next(e for e in ctx.ui.elements if e["text"] == text)


@pytest.mark.parametrize("action", ["click", "long_press", "type"])
def test_indexed_actions_avoid_later_touchable_sibling(action):
    ctx = context([node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)])
    index = element(ctx, "target")["index"]
    if action == "click":
        result = asyncio.run(click(index, ctx=ctx))
    elif action == "long_press":
        result = asyncio.run(long_press(index, ctx=ctx))
    else:
        result = asyncio.run(type_text("test", index=index, ctx=ctx))
        ctx.driver.input_text.assert_awaited_once_with("test", False)
    assert result.success
    operation = ctx.driver.swipe if action == "long_press" else ctx.driver.tap
    x, y = operation.await_args.args[:2]
    assert 0 <= x < 1000 and 0 <= y < 400
    assert not 300 <= x < 700


def test_drawing_order_can_block_an_element_with_a_higher_index():
    ctx = context([node("overlay", (300, 0, 700, 400), drawingOrder=8), node("target")])
    target = element(ctx, "target")
    assert target["index"] > element(ctx, "overlay")["index"]
    result = asyncio.run(click(target["index"], ctx=ctx))
    assert result.success
    assert not 300 <= ctx.driver.tap.await_args.args[0] < 700


def test_fully_covered_target_fails_without_dispatching_a_tap():
    ctx = context([node("target"), node("overlay", drawingOrder=2)])
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert not result.success
    assert "No clear tap point" in result.summary
    ctx.driver.tap.assert_not_awaited()


def test_child_filled_container_remains_clickable_and_children_keep_indices():
    ctx = context(
        [
            node(
                "row",
                children=[
                    node("icon", (0, 0, 200, 400), isClickable=False),
                    node("label", (200, 0, 1000, 400), isClickable=False, drawingOrder=2),
                ],
            )
        ]
    )
    result = asyncio.run(click(element(ctx, "row")["index"], ctx=ctx))
    assert result.success
    ctx.driver.tap.assert_awaited_once_with(500, 200)
    assert [e["index"] for e in ctx.ui.elements] == [1, 2, 3, 4]
    assert all(e["children"] == [] for e in ctx.ui.elements)


def test_independent_child_and_parent_keep_their_own_tap_targets():
    ctx = context([node("row", children=[node("child", (800, 0, 1000, 400))])])
    for text, expected in [("row", (500, 200)), ("child", (900, 200))]:
        result = asyncio.run(click(element(ctx, text)["index"], ctx=ctx))
        assert result.success
        assert ctx.driver.tap.await_args.args == expected


def test_stealth_jitter_cannot_reenter_known_obstructions(monkeypatch):
    monkeypatch.setattr("mobilerun.core.state.stealth.random.randint", lambda a, b: 0)
    ctx = context(
        [node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)],
        stealth=True,
    )
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert result.success
    assert not 300 <= ctx.driver.tap.await_args.args[0] < 700


def test_multiple_siblings_can_collectively_cover_target():
    ctx = context(
        [
            node("target"),
            node("left", (0, 0, 500, 400), drawingOrder=2),
            node("right", (500, 0, 1000, 400), drawingOrder=3),
        ],
        stealth=True,
    )
    result = asyncio.run(click(element(ctx, "target")["index"], ctx=ctx))
    assert not result.success
    ctx.driver.tap.assert_not_awaited()


@pytest.mark.parametrize("stealth", [False, True])
@pytest.mark.parametrize("action", ["click", "long_press", "type"])
def test_normalized_indexed_actions_tap_device_pixels(action, stealth):
    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
        stealth=stealth,
    )
    assert "(861,895,972,945)" in ctx.ui.formatted_text
    index = element(ctx, "send")["index"]
    if action == "click":
        result = asyncio.run(click(index, ctx=ctx))
    elif action == "long_press":
        result = asyncio.run(long_press(index, ctx=ctx))
    elif action == "type":
        result = asyncio.run(type_text("test", index=index, ctx=ctx))
    assert result.success
    operation = ctx.driver.swipe if action == "long_press" else ctx.driver.tap
    x, y = operation.await_args.args[:2]
    assert 930 <= x < 1050 and 2150 <= y < 2270
    if not stealth:
        assert (x, y) == (990, 2210)
        if action == "click":
            assert result.summary.endswith("Coordinates: (916, 920)")
        if action == "long_press":
            assert result.summary.endswith("at (916, 920)")


def test_normalized_macro_records_device_pixels():
    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
    )
    ctx.macro_recorder = SimpleNamespace(record_action=Mock())
    asyncio.run(click(element(ctx, "send")["index"], ctx=ctx))
    action = ctx.macro_recorder.record_action.call_args.args[0]
    assert action == {"action_type": "tap", "x": 990, "y": 2210}

"""Regression coverage through raw Portal data, formatting, and real actions."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from mobilerun.core.state.android import AndroidStateProvider
from mobilerun.core.state.filters import ConciseFilter
from mobilerun.core.state.formatters import IndexedFormatter

from tests.unit.device_fixtures import MemoryDevice, device_turn, execute_tool


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

    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            await self.tap_mock(x, y)

    driver = Phone(supported_operations=frozenset({"tap"}))
    driver._get_ui_tree = AsyncMock(return_value=raw)
    driver.tap_mock = AsyncMock()
    provider = AndroidStateProvider(driver, ConciseFilter(), IndexedFormatter(), **provider_kwargs)
    ui = asyncio.run(provider.get_state())
    return device_turn(driver, ui, provider)


def element(ctx, text):
    return next(e for e in ctx.state.current.ui.elements if e["text"] == text)


async def click(ctx, index):
    return await execute_tool(ctx, "tap_element", index=index)


def test_indexed_actions_avoid_later_touchable_sibling():
    ctx = context([node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)])
    index = element(ctx, "target")["index"]
    result = asyncio.run(click(ctx, index))
    assert result.success
    x, y = ctx.device.tap_mock.await_args.args[:2]
    assert 0 <= x < 1000 and 0 <= y < 400
    assert not 300 <= x < 700


def test_drawing_order_can_block_an_element_with_a_higher_index():
    ctx = context([node("overlay", (300, 0, 700, 400), drawingOrder=8), node("target")])
    target = element(ctx, "target")
    assert target["index"] > element(ctx, "overlay")["index"]
    result = asyncio.run(click(ctx, target["index"]))
    assert result.success
    assert not 300 <= ctx.device.tap_mock.await_args.args[0] < 700


def test_fully_covered_target_fails_without_dispatching_a_tap():
    ctx = context([node("target"), node("overlay", drawingOrder=2)])
    result = asyncio.run(click(ctx, element(ctx, "target")["index"]))
    assert not result.success
    assert "ValueError" in result.summary
    ctx.device.tap_mock.assert_not_awaited()


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
    result = asyncio.run(click(ctx, element(ctx, "row")["index"]))
    assert result.success
    ctx.device.tap_mock.assert_awaited_once_with(500, 200)
    assert [e["index"] for e in ctx.state.current.ui.elements] == [1, 2, 3, 4]
    assert all(e["children"] == [] for e in ctx.state.current.ui.elements)


def test_independent_child_and_parent_keep_their_own_tap_targets():
    ctx = context([node("row", children=[node("child", (800, 0, 1000, 400))])])
    for text, expected in [("row", (500, 200)), ("child", (900, 200))]:
        result = asyncio.run(click(ctx, element(ctx, text)["index"]))
        assert result.success
        assert ctx.device.tap_mock.await_args.args == expected


def test_stealth_jitter_cannot_reenter_known_obstructions(monkeypatch):
    monkeypatch.setattr("mobilerun.core.state.stealth.random.randint", lambda a, b: 0)
    ctx = context(
        [node("target"), node("overlay", (300, 0, 700, 400), drawingOrder=2)],
        stealth=True,
    )
    result = asyncio.run(click(ctx, element(ctx, "target")["index"]))
    assert result.success
    assert not 300 <= ctx.device.tap_mock.await_args.args[0] < 700


def test_multiple_siblings_can_collectively_cover_target():
    ctx = context(
        [
            node("target"),
            node("left", (0, 0, 500, 400), drawingOrder=2),
            node("right", (500, 0, 1000, 400), drawingOrder=3),
        ],
        stealth=True,
    )
    result = asyncio.run(click(ctx, element(ctx, "target")["index"]))
    assert not result.success
    ctx.device.tap_mock.assert_not_awaited()


@pytest.mark.parametrize("stealth", [False, True])
def test_normalized_indexed_actions_tap_device_pixels(stealth):
    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
        stealth=stealth,
    )
    assert "(861,895,972,945)" in ctx.state.current.ui.formatted_text
    index = element(ctx, "send")["index"]
    result = asyncio.run(click(ctx, index))
    assert result.success
    x, y = ctx.device.tap_mock.await_args.args[:2]
    assert 930 <= x < 1050 and 2150 <= y < 2270
    if not stealth:
        assert (x, y) == (990, 2210)


def test_normalized_macro_records_device_pixels():
    from llama_index.core.tools import ToolSelection

    from mobilerun_agents.agent.checkpoints.macro_export import macro_data
    from mobilerun_agents.agent.workflows.core.state import ActionRecord

    ctx = context(
        [node("send", (930, 2150, 1050, 2270))],
        screen=(1080, 2400),
        use_normalized=True,
    )
    call = ToolSelection(
        tool_id="tap", tool_name="tap_element", tool_kwargs={"index": element(ctx, "send")["index"]}
    )
    result = asyncio.run(ctx.registry.execute(call, ctx.ctx))
    record = ActionRecord(
        step=1,
        call=ctx.registry.resolve(call),
        ok=result.success,
        summary=result.summary,
        before=ctx.state.current,
    )
    [action] = macro_data("Send", [record], ctx.registry.redactor)["actions"]
    assert (action["action_type"], action["x"], action["y"]) == ("tap", 990, 2210)

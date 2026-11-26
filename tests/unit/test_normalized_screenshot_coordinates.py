import asyncio
from io import BytesIO

import pytest
from mobilerun.core.devices import Device
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.core.state.ui_state import UIState
from PIL import Image

from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.user_config import ToolsConfig
from tests.unit.device_fixtures import MemoryDevice, tool_run


def _png(width: int = 1080, height: int = 2400) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=(16, 20, 24)).save(output, format="PNG")
    return output.getvalue()


class _Driver(Device):
    platform = "Android"
    capabilities = set()
    supported_buttons = set()

    def __init__(self) -> None:
        super().__init__()
        self.taps = []
        self.swipes = []

    @property
    def supported_operations(self):
        return {"tap", "swipe"}

    async def screenshot(self) -> bytes:
        return _png()

    async def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000) -> None:
        self.swipes.append((x1, y1, x2, y2))

    async def _get_date(self) -> str:
        return "2026-09-29"

    def _create_state_provider(self, options=None):
        from mobilerun.core.state.options import (
            StateProviderOptions,
            android_state_provider,
        )

        return android_state_provider(self, options or StateProviderOptions())


def _normalized_state():
    driver = _Driver()
    provider = ScreenshotOnlyStateProvider(driver, use_normalized=True)
    return driver, provider, asyncio.run(provider.get_state())


def _ctx():
    driver, provider, state = _normalized_state()
    return (
        tool_run(
            config=ToolsConfig(),
            driver=driver,
            ui=state,
            state_provider=provider,
            macro_recorder=None,
        ),
        driver,
    )


async def click_at(ctx, x, y):
    return (await build_registry(ctx, [], None).by_name["tap"].acall(x=x, y=y)).raw_output


async def swipe(ctx, start, end):
    return (
        await build_registry(ctx, [], None)
        .by_name["swipe"]
        .acall(x1=start[0], y1=start[1], x2=end[0], y2=end[1])
    ).raw_output


def test_base_normalization_does_not_rescale_device_actions() -> None:
    ctx, driver = _ctx()

    tap = asyncio.run(click_at(ctx, 900, 920))
    drag = asyncio.run(swipe(ctx, [500, 800], [500, 200]))

    assert driver.taps == [(900, 920)]
    assert driver.swipes == [(500, 800, 500, 200)]
    assert tap.success and drag.success


def test_pixel_action_summaries_report_device_pixels() -> None:
    driver = _Driver()
    provider = ScreenshotOnlyStateProvider(driver)
    state = asyncio.run(provider.get_state())
    ctx = tool_run(
        config=ToolsConfig(), driver=driver, ui=state, state_provider=provider, macro_recorder=None
    )

    result = asyncio.run(click_at(ctx, 461, 1024))

    assert "'x': 461, 'y': 1024" in result.summary
    assert driver.taps[0] == (461, 1024)


def test_normalized_clamp_only_moves_the_1000_endpoint() -> None:
    state = UIState(
        elements=[],
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=1080,
        screen_height=2400,
        use_normalized=True,
    )

    assert state.convert_point(1000, 999) == (1079, 2397)
    assert state.convert_point(1200, 1100) == (1296, 2640)


@pytest.mark.parametrize("point", [(1080, 500), (500, -1)])
def test_normalized_actions_reject_points_outside_device(point) -> None:
    ctx, driver = _ctx()

    result = asyncio.run(click_at(ctx, *point))

    assert result.success is False
    assert "1080x2400" in result.summary
    assert driver.taps == []


def _a11y_context(vision: bool, screen=(1080, 2400)):
    from unittest.mock import AsyncMock

    from mobilerun.core.state.android import AndroidStateProvider
    from mobilerun.core.state.filters import ConciseFilter
    from mobilerun.core.state.formatters import IndexedFormatter

    def node(text, bounds, children=()):
        return {
            "text": text,
            "className": "android.widget.Button",
            "boundsInScreen": dict(zip(("left", "top", "right", "bottom"), bounds, strict=True)),
            "isClickable": True,
            "isVisibleToUser": True,
            "children": list(children),
        }

    raw = {
        "a11y_tree": node("root", (0, 0, 1080, 2400), [node("Send", (930, 2150, 1050, 2270))]),
        "phone_state": {},
        "device_context": {"screen_bounds": dict(zip(("width", "height"), screen, strict=True))},
    }

    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            await self.tap_mock(x, y)

        async def swipe(
            self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
        ) -> None:
            await self.swipe_mock(x1, y1, x2, y2, duration_ms)

    driver = Phone(supported_operations=frozenset({"tap", "swipe"}))
    driver._get_ui_tree = AsyncMock(return_value=raw)
    driver.tap_mock = AsyncMock()
    driver.swipe_mock = AsyncMock()
    provider = AndroidStateProvider(
        driver,
        ConciseFilter(),
        IndexedFormatter(),
        use_normalized=True,
        vision_enabled=vision,
    )
    ui = asyncio.run(provider.get_state())
    return tool_run(
        config=ToolsConfig(), ui=ui, driver=driver, state_provider=provider, macro_recorder=None
    )


def test_normalized_a11y_click_at_rejects_points_outside_device() -> None:
    ctx = _a11y_context(vision=True)

    inside = asyncio.run(click_at(ctx, 1079, 2399))
    outside = asyncio.run(click_at(ctx, 1080, 5))
    drag = asyncio.run(swipe(ctx, [500, 800], [500, -1]))

    assert inside.success
    ctx.device.tap_mock.assert_awaited_once_with(1079, 2399)
    assert outside.success is False
    assert "1080x2400" in outside.summary
    assert drag.success is False
    ctx.device.swipe_mock.assert_not_awaited()


def test_normalized_a11y_refuses_coordinates_without_a_screen_size() -> None:
    ctx = _a11y_context(vision=True, screen=(0, 0))
    assert "(normalized [0-1000])" not in ctx.ui.formatted_text

    tap = asyncio.run(click_at(ctx, 500, 500))
    drag = asyncio.run(swipe(ctx, [500, 800], [500, 200]))

    assert tap.success is False and drag.success is False
    assert "unavailable for this step" in tap.summary
    ctx.device.tap_mock.assert_not_awaited()
    ctx.device.swipe_mock.assert_not_awaited()

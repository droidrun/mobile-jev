import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from mobilerun.config.prompts import PromptLoader
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.core.state.ui_state import UIState

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.tools.actions import (
    click_area,
    click_at,
    long_press_at,
    swipe,
)
from mobilerun.agents.tools.signatures import build_tool_registry
from mobilerun.agents.workflows.droid.droid_agent import (
    MobileAgent,
    _effective_disabled_tools,
)


def _png(width: int = 1320, height: int = 2868) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    )


class ScreenshotOnlyStateProviderTest(unittest.TestCase):

    def _render_fast_agent_prompt(self, *, screenshot_only: bool) -> str:
        repo = Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
        template = (repo / "assets/prompts/fast_agent/system.jinja2").read_text()
        return PromptLoader.render_template(
            template,
            {
                "tool_descriptions": "",
                "available_secrets": [],
                "available_tools": set(),
                "variables": {},
                "output_schema": None,
                "parallel_tools": False,
                "vision": True,
                "platform": "android",
                "screenshot_only": screenshot_only,
            },
        )

    def test_fast_agent_prompt_describes_screenshot_only_state(self):
        prompt = self._render_fast_agent_prompt(screenshot_only=True)

        self.assertNotIn(
            "list of all currently visible UI elements with their indices",
            prompt,
        )
        self.assertIn("Screenshot-only device state", prompt)
        self.assertIn("There is no accessibility tree", prompt)
        self.assertIn("indexed UI element list", prompt)

    def test_fast_agent_prompt_keeps_indexed_state_for_normal_mode(self):
        prompt = self._render_fast_agent_prompt(screenshot_only=False)

        self.assertIn(
            "list of all currently visible UI elements with their indices",
            prompt,
        )

    def test_tool_filter_keeps_coordinate_tools_and_direct_text(self):
        class FakeDriver:
            async def screenshot(self):
                return _png()

        provider = ScreenshotOnlyStateProvider(FakeDriver())

        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="ios",
                screenshot_only=True,
            )
            capabilities = {
                "tap",
                "swipe",
                "input_text",
                "press_button",
                "screenshot",
            } | provider.capabilities
            registry.disable_unsupported(capabilities)
            registry.disable(
                _effective_disabled_tools(
                    ["click_at", "click_area", "long_press_at"],
                    provider,
                )
            )
            return registry

        registry = asyncio.run(run())

        self.assertIn("click_at", registry.tools)
        self.assertIn("click_area", registry.tools)
        self.assertIn("long_press_at", registry.tools)
        self.assertIn("swipe", registry.tools)
        self.assertIn("type_text", registry.tools)
        self.assertIn("system_button", registry.tools)
        self.assertNotIn("click", registry.tools)
        self.assertNotIn("type", registry.tools)

    def test_screenshot_only_coordinate_tools_describe_model_screenshot_coordinates(
        self,
    ):
        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="ios",
                screenshot_only=True,
            )
            return registry

        registry = asyncio.run(run())

        self.assertIn(
            "screenshot pixel coordinates",
            registry.tools["click_at"].description,
        )
        self.assertIn("grid is only a reference", registry.tools["click_at"].description)
        self.assertIn("do not use grid-cell numbers", registry.tools["click_at"].description)
        self.assertIn("Prefer click_at", registry.tools["click_at"].description)
        self.assertIn("coordinate grid", registry.tools["click_at"].description)
        self.assertIn(
            "large, unambiguous targets",
            registry.tools["click_area"].description,
        )
        self.assertIn(
            "prefer click_at",
            registry.tools["click_area"].description,
        )
        self.assertIn(
            "screenshot pixel coordinates",
            registry.tools["long_press_at"].description,
        )
        self.assertIn("screenshot coordinate", registry.tools["swipe"].description)

        for name in ("click_at", "click_area", "long_press_at", "swipe"):
            description = registry.tools[name].description
            self.assertNotIn("0..1000", description)
            self.assertNotIn("destructive controls", description)
            self.assertNotIn("Do not tap toggles", description)

    def test_normal_coordinate_tools_do_not_describe_screenshot_only_mode(self):
        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="ios",
                screenshot_only=False,
            )
            return registry

        registry = asyncio.run(run())

        for name in ("click_at", "click_area", "long_press_at", "swipe"):
            description = registry.tools[name].description
            self.assertNotIn("screenshot-only", description)
            self.assertNotIn("screenshot pixel", description)
            self.assertNotIn("coordinate grid", description)
            self.assertNotIn("grid-cell", description)
            self.assertNotIn("destructive controls", description)
            self.assertNotIn("Do not tap toggles", description)

        self.assertIn("Click at position", registry.tools["click_at"].description)
        self.assertIn("Duration is in seconds", registry.tools["swipe"].description)

    def test_screenshot_only_agent_sources_resize_screenshots(self):
        repo = Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
        agent_files = [
            repo / "workflows/fast_agent/fast_agent.py",
            repo / "workflows/manager/manager_agent.py",
            repo / "workflows/manager/stateless_manager_agent.py",
            repo / "workflows/executor/executor_agent.py",
        ]

        for path in agent_files:
            self.assertIn("_resize_model_screenshot_async", path.read_text())

    def test_visual_remote_exact_app_launch_tool_needs_only_start_app(self):
        async def run():
            registry, _ = await build_tool_registry(
                supported_buttons={"enter"},
                platform="android",
                exact_app_launch=True,
            )
            registry.disable_unsupported({"start_app"})
            return registry

        registry = asyncio.run(run())

        self.assertIn("open_app", registry.tools)
        self.assertEqual(
            registry.tools["open_app"].params,
            {"app_id": {"type": "string", "required": True}},
        )
        self.assertEqual(registry.tools["open_app"].deps, {"start_app"})
        self.assertIn("exact app identifier", registry.tools["open_app"].description)


class ScreenshotOnlyCoordinateValidationTest(unittest.TestCase):
    def _ctx(self):
        from tests.unit.device_fixtures import MemoryDevice

        class FakeDriver(MemoryDevice):
            def __init__(self):
                super().__init__(supported_operations=frozenset({"tap", "swipe"}))
                self.taps = []
                self.swipes = []

            async def tap(self, x: int, y: int) -> None:
                self.taps.append((x, y))

            async def swipe(
                self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
            ) -> None:
                self.swipes.append((x1, y1, x2, y2, duration_ms))

        driver = FakeDriver()
        ui = UIState(
            elements=[],
            formatted_text="",
            focused_text="",
            phone_state={},
            screen_width=1000,
            screen_height=2000,
            use_normalized=False,
        )
        provider = SimpleNamespace(
            requires_coordinate_tools=True, requires_active_contract_for_coords=False
        )
        return (
            SimpleNamespace(driver=driver, ui=ui, state_provider=provider, macro_recorder=None),
            driver,
        )

    def test_click_at_valid_coordinate_converts_and_taps(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_at(500, 500, ctx=ctx))

        self.assertTrue(result.success)
        self.assertEqual(driver.taps, [(500, 500)])

    def test_click_at_rejects_out_of_range_before_tapping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_at(1000, 2000, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.taps, [])

    def test_swipe_rejects_out_of_range_before_swiping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(swipe([500, 2000], [500, 500], ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.swipes, [])

    def test_click_area_rejects_out_of_range_corner_before_tapping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(click_area(100, 100, 1000, 300, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.taps, [])

    def test_long_press_at_rejects_out_of_range_before_swiping(self):
        ctx, driver = self._ctx()

        result = asyncio.run(long_press_at(-1, 500, ctx=ctx))

        self.assertFalse(result.success)
        self.assertIn("Coordinates must be inside the screenshot size 1000x2000", result.summary)
        self.assertEqual(driver.swipes, [])


class VisualRemoteConfigTest(unittest.TestCase):

    def test_mobile_agent_forces_visual_remote_screenshot_vision(self):
        config = MobileConfig.from_dict(
            {
                "agent": {"name": "external-agent"},
                "device": {"control_backend": "visual-remote"},
            }
        )

        agent = MobileAgent("Check Wi-Fi", config=config)

        self.assertTrue(agent.config.agent.vision_only)
        self.assertTrue(agent.config.agent.manager.vision)
        self.assertTrue(agent.config.agent.executor.vision)
        self.assertTrue(agent.config.agent.fast_agent.vision)


if __name__ == "__main__":
    unittest.main()

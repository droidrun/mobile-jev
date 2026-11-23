"""Exercise the runner and real agent workflows with local fakes."""

import asyncio
import importlib
import json
import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms.mock import MockLLM

from mobilerun.agents import runner
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.logs.cli import _setup_cli_logging
from mobilerun.agents.tools.action_result import ActionResult
from mobilerun.agents.tools.tool_registry import ToolRegistry
from tests.unit.device_fixtures import MemoryDevice


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    ("success", "message", "last_lines", "level"),
    [
        (True, "done", ["✓ Done in 3 steps", "done"], "normal"),
        (
            False,
            "Battery unavailable",
            ["✗ Failed after 3 steps: Battery unavailable"],
            "normal",
        ),
        pytest.param(
            True,
            "Number 7 is skipped",
            ["✓ Done in 3 steps", "Number 7 is skipped"],
            "normal",
        ),
        (
            True,
            "Number 7 is skipped",
            ["✓ Done in 3 steps", "Number 7 is skipped"],
            "quiet",
        ),
        (True, "", ["✓ Done in 3 steps"], "normal"),
    ],
)
def test_real_agent_normal_transcript(
    monkeypatch, capsys, success, message, last_lines, level, batched
):
    droid = importlib.import_module("mobilerun.agents.workflows.droid.droid_agent")
    fast = importlib.import_module("mobilerun.agents.workflows.fast_agent.fast_agent")
    config = MobileConfig()
    config.telemetry.enabled = False
    config.agent.reasoning = False
    config.agent.streaming = False
    config.agent.fast_agent.vision = False
    config.agent.fast_agent.tool_calling = "auto"
    config.agent.after_sleep_action = 0
    config.device.serial = None
    config.tools.disabled_tools = []
    monkeypatch.setattr(runner.ConfigLoader, "load", lambda path: config)

    class FakeLLM(MockLLM):
        supports_native_tool_calls: ClassVar[bool] = True
        native_tool_api: ClassVar[str] = "responses"

    monkeypatch.setattr(runner, "load_llm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(droid, "capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(droid, "setup_tracing", lambda *args, **kwargs: None)
    monkeypatch.setattr(fast, "get_usage_from_response", lambda *args: None)

    class Device(MemoryDevice):
        identifier = "emulator-5554"
        platform = "android"
        supported_operations = set()
        supported_buttons = set()
        credentials = []
        recording = None

        async def _get_date(self):
            return ""

        async def tap(self, x, y):
            pass

        async def screenshot(self):
            return None

    class StateProvider:
        capabilities = set()
        use_normalized = False
        requires_coordinate_tools = False
        screenshot_matches_input_coords = False

        async def get_state(self):
            return SimpleNamespace(
                formatted_text="Settings", focused_text="", elements=[], phone_state={}
            )

        async def get_final_state(self):
            return await self.get_state()

    async def build_registry(**kwargs):
        registry = ToolRegistry()

        async def tap(x, y, *, ctx):
            await ctx.driver.tap(x, y)
            return ActionResult(success=True, summary="ok")

        async def complete(success, message, *, ctx):
            await ctx.shared_state.complete(success, message=message)
            return ActionResult(success=True, summary=message)

        registry.register("tap", tap, {"x": {"type": "number"}, "y": {"type": "number"}}, "Tap")
        registry.register(
            "complete",
            complete,
            {
                "success": {"type": "boolean"},
                "message": {"type": "string"},
            },
            "Done",
        )
        return registry, {"tap", "complete"}

    monkeypatch.setattr(droid, "build_tool_registry", build_registry)
    real_agent = droid.MobileAgent
    monkeypatch.setattr(
        runner,
        "MobileAgent",
        lambda **kwargs: real_agent(state_provider=StateProvider(), **kwargs),
    )

    replies = iter(
        [
            ("thinking one", "tap", {"x": 1, "y": 2}),
            ("thinking two", "tap", {"x": 3, "y": 4}),
            ("thinking three", "tap", {"x": 5, "y": 6}),
            (
                "finished",
                "complete",
                {
                    "success": success,
                    "message": message,
                },
            ),
        ]
    )

    async def respond(llm, messages, *, stream, **kwargs):
        text, name, args = next(replies)
        blocks = [
            TextBlock(text=text),
            ToolCallBlock(tool_call_id=name, tool_name=name, tool_kwargs=json.dumps(args)),
        ]
        if batched and name == "tap" and args["x"] == 1:
            _, next_name, next_args = next(replies)
            blocks.append(
                ToolCallBlock(
                    tool_call_id="tap2",
                    tool_name=next_name,
                    tool_kwargs=json.dumps(next_args),
                )
            )
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                blocks=blocks,
            )
        )

    monkeypatch.setattr(fast, "acall_with_retries", respond)
    logger = logging.getLogger("mobilerun")
    handlers, old_level, propagate = logger.handlers[:], logger.level, logger.propagate
    try:
        result = asyncio.run(
            runner.run_agent(
                "Go to battery",
                device=Device(),
                provider="fake",
                model="fake",
                log_level=level,
                on_config=lambda c, d: _setup_cli_logging(d, c.logging.level),
            )
        )
        transcript = capsys.readouterr().err
        print("CAPTURED:", repr(transcript))
        assert result.success is success
        action_lines = [
            "Goal: Go to battery",
            "emulator-5554 · step 1  tap (1, 2)",
            "emulator-5554 · step 2  tap (3, 4)",
            "emulator-5554 · step 3  tap (5, 6)",
        ]
        assert transcript.splitlines() == (action_lines if level == "normal" else []) + last_lines
    finally:
        logger.handlers = handlers
        logger.setLevel(old_level)
        logger.propagate = propagate

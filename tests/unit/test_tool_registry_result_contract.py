import asyncio

import pytest

from mobilerun.agents.tools.action_result import ActionResult
from mobilerun.agents.tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Failed to act", ActionResult(success=False, summary="Failed to act")),
        ((True, "done"), ActionResult(success=True, summary="done")),
        (None, ActionResult(success=True, summary="Done")),
        (ActionResult(success=False, summary="no"), ActionResult(False, "no")),
    ],
)
def test_registered_tool_normalizes_result(value, expected):
    registry = ToolRegistry()
    registry.register("legacy", lambda **kwargs: value, {}, "legacy")

    result = asyncio.run(registry.tools["legacy"].fn(ctx=None))

    assert result == expected

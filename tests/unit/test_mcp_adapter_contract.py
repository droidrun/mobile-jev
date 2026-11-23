import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mobilerun.agents.mcp.adapter import _create_tool_wrapper


def test_mcp_wrapper_ignores_text_attribute_on_non_text_blocks():
    result = SimpleNamespace(content=[SimpleNamespace(type="image", text="not text content")])
    manager = SimpleNamespace(call_tool=AsyncMock(return_value=result))
    wrapper = _create_tool_wrapper("sample", manager)
    assert asyncio.run(wrapper()) == str(result)

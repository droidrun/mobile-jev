import inspect

from llama_index.core.tools import ToolSelection

from mobilerun_agents.agent.action_selector import base


def test_selector_accepts_context_device_memory_and_registry():
    assert tuple(inspect.signature(base.ActionSelector.select).parameters) == (
        "self",
        "ctx",
        "device",
        "memory",
        "registry",
    )
    assert not hasattr(base.ActionSelector, "bind")
    call = ToolSelection(tool_id="1", tool_name="wait", tool_kwargs={})
    assert base.Selection(calls=[call]).calls == [call]

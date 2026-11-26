import asyncio
import socket
from contextlib import asynccontextmanager
from io import BytesIO
from unittest.mock import AsyncMock, NonCallableMock

import pytest
from click.testing import CliRunner
from llama_index.core.llms import ChatMessage, ChatResponse
from mobilerun.cli.app import cli
from mobilerun.cli.devices.base.options import DeviceTarget
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from PIL import Image

from mobilerun_agents import AgentOperator, LlamaIndexPlanner, Run, tool
from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig
from tests.unit.device_fixtures import MemoryDevice


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args):
        pytest.fail("Debug logging tests must not connect to a network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.mark.parametrize(
    "debug,vision,plan",
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True)],
)
def test_debug_run(debug, vision, plan, monkeypatch):
    @tool(sensitive_parameters=frozenset({"value"}))
    async def unlock(run: Run, value: str) -> str:
        """Unlock the fake session."""
        return value

    phone = MemoryDevice()
    provider = NonCallableMock(
        spec=StateProvider, capabilities=set(), requires_coordinate_tools=False
    )
    provider.get_state = AsyncMock(
        return_value=UIState([], "ui", "", {"packageName": "app", "activityName": "Main"}, 100, 100)
    )
    phone._set_state_provider(provider)
    image = BytesIO()
    Image.new("RGB", (24, 40)).save(image, "PNG")
    phone.screenshot = AsyncMock(return_value=image.getvalue())

    @asynccontextmanager
    async def opened(target):
        yield phone

    cfg = MobileConfig(
        llm_profiles={
            "default": LLMProfile(
                provider="anthropic", model="claude-sonnet-4-5", vision="on" if vision else "off"
            )
        }
    )
    cfg.analytics.enabled = cfg.checkpoints.enabled = False
    monkeypatch.setattr(ConfigLoader, "load", lambda *_: cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-not-a-key")
    replies = iter(
        [
            "not an invocation",
            '<function_calls><invoke name="unlock"><parameter name="value">private-input</parameter></invoke></function_calls>',
            "Verify   the\n task. private-input fixture-not-a-key <add_memory>Checked</add_memory>"
            + "x" * 300
            + 'TRUNCATED<function_calls><invoke name="complete"><parameter name="success">true</parameter><parameter name="message">Finished</parameter></invoke></function_calls>',
        ]
    )
    if plan:
        from tests.unit.agent.workflows.test_planning import PlannerLLM

        planner = PlannerLLM(
            ("set_plan", {"subgoals": ["Open app", "Read screen"]}),
            ("complete", {"success": True, "message": "Finished"}),
        )
        monkeypatch.setattr(
            "mobilerun_agents.cli.agent.run.default_planner",
            AsyncMock(return_value=LlamaIndexPlanner(planner)),
        )
        complete = '<function_calls><invoke name="complete"><parameter name="success">true</parameter><parameter name="message">Finished</parameter></invoke></function_calls>'
        replies = iter([complete])

    async def chat(self, messages, **kwargs):
        return ChatResponse(
            message=ChatMessage(role="assistant", content=next(replies)),
            raw={
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 3,
                    "total_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 2},
                }
            },
        )

    monkeypatch.setattr(DeviceTarget, "open", opened)
    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    monkeypatch.setattr(
        "mobilerun_agents.cli.agent.run.AgentOperator",
        lambda **kw: AgentOperator(**kw, tools=[unlock]),
    )
    outcome = CliRunner().invoke(
        cli,
        (["--debug"] if debug else [])
        + ["agent", "run", "unlock", "serial"]
        + (["--plan"] if plan else []),
        env={"COLUMNS": "500"},
    )
    assert outcome.exit_code == 0, outcome.exception
    if plan:
        assert "LLM: unknown" in outcome.output
        assert "planning: executor: llm:anthropic" in outcome.output
        assert "planning: subgoals: Open app; Read screen" in outcome.output
        assert "planning: reasoning: Review progress." in outcome.output
        assert "TRUNCATED" not in outcome.output
        return
    assert "private-input" not in outcome.output and "TRUNCATED" not in outcome.output
    assert "fixture-not-a-key" not in outcome.output
    assert "[REDACTED]" not in outcome.output
    assert "1 · unlock *** ✓" in outcome.output and "2 · complete ✓" in outcome.output
    for detail in [
        "  screen: app/Main",
        "24×40" if vision else "no image",
        "  selector: llm:anthropic/claude-sonnet-4-5",
        "  thought: Verify the task.",
        "  memory: Checked",
        '"value": "***"',
        "unlock → unlock",
        "input=4 output=3 reasoning=2",
        "  repair 1:",
        "  result: success · ***",
        "total tokens=21",
    ]:
        assert (detail in outcome.output) == debug, detail
    if debug:
        remaining = outcome.output
        for marker in (
            "1 · attempt 1",
            "  screen:",
            "  selector:",
            "  image:",
            "  LLM:",
            "  repair 1:",
            "1 · attempt 2",
            "  screen:",
            "  LLM:",
            "1 · unlock *** ✓",
            "  selection:",
            "  result:",
            "2 · attempt 1",
            "  thought:",
            "  memory:",
            "2 · complete ✓",
            "  selection:",
            "  result:",
        ):
            _, found, remaining = remaining.partition(marker)
            assert found, marker
        assert "add_memory" not in outcome.output and "run folder=" not in outcome.output
        thought = next(line for line in outcome.output.splitlines() if "  thought:" in line)
        assert thought.startswith("  thought: Verify the task. *** *** x") and thought.endswith("…")


@pytest.mark.parametrize(
    "primary_action,reason", [("delegate", "explicit delegate"), ("wait", "confidence 0.4 < 0.8")]
)
def test_delegation_records_the_child_and_reason(primary_action, reason):
    from mobilerun_agents.agent.action_selector import (
        ActionSelection,
        ActionSelector,
        DelegatingSelector,
    )
    from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
    from mobilerun_agents.agent.logs.diagnostics import sink
    from tests.unit.agent.selector_fixtures import selection_run
    from tests.unit.device_fixtures import fake_tools

    class Script(ActionSelector):
        def __init__(self, label, action):
            self.debug_label, self.action = label, action

        async def select(self, run):
            return ActionSelection(actions=[ToolCall(self.action)], confidence=0.4)

    events = []
    token = sink.set(events.append)
    try:
        selector = DelegatingSelector(
            Script("jev", primary_action), Script("llm:fixture", "wait"), min_confidence=0.8
        )
        asyncio.run(selector.select(selection_run("wait", tools=fake_tools("delegate", "wait"))))
    finally:
        sink.reset(token)
    assert [event.text for event in events] == [f"jev → llm:fixture ({reason})"]

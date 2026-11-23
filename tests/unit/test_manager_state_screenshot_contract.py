import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.manager import manager_agent, stateless_manager_agent
from mobilerun.agents.workflows.manager.events import ManagerContextEvent


@pytest.mark.parametrize(
    "cls", [manager_agent.ManagerAgent, stateless_manager_agent.StatelessManagerAgent]
)
def test_missing_shared_state_gets_a_fresh_state(cls):
    def create():
        return cls(llm=None, action_ctx=None, state_provider=None, agent_config=AgentConfig())

    first, second = create(), create()
    assert first.shared_state.instruction == ""
    assert first.shared_state is not second.shared_state

    shared = MobileAgentState(instruction="shared")
    configured = cls(
        llm=None,
        action_ctx=None,
        state_provider=None,
        agent_config=AgentConfig(),
        shared_state=shared,
    )
    assert configured.shared_state is shared


@pytest.mark.parametrize(
    "module,cls",
    [
        (manager_agent, manager_agent.ManagerAgent),
        (stateless_manager_agent, stateless_manager_agent.StatelessManagerAgent),
    ],
)
def test_vision_converts_screenshot_path_to_bytes_before_resize(module, cls, tmp_path, monkeypatch):
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (2, 2)).save(path)
    state = MobileAgentState(screenshot=str(path))
    agent = cls(
        llm=SimpleNamespace(class_name=lambda: "fake"),
        action_ctx=None,
        state_provider=SimpleNamespace(),
        agent_config=AgentConfig(),
        shared_state=state,
    )
    agent.vision = True
    agent._build_system_prompt = AsyncMock(return_value="prompt")
    agent._build_prompt = AsyncMock(return_value="prompt")
    agent._validate_and_retry = AsyncMock(return_value="response")
    resize = AsyncMock(side_effect=lambda provider, image: image)
    monkeypatch.setattr(module, "_resize_model_screenshot_async", resize)
    monkeypatch.setattr(
        module,
        "acall_with_retries",
        AsyncMock(return_value=SimpleNamespace(message=SimpleNamespace(content="response"))),
    )
    monkeypatch.setattr(module, "get_usage_from_response", lambda *args: None)
    asyncio.run(
        agent.get_response(
            SimpleNamespace(write_event_to_stream=lambda event: None),
            ManagerContextEvent(),
        )
    )
    assert resize.await_args.args[1].startswith(b"\x89PNG")

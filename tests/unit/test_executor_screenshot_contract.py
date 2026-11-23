import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llama_index.core.workflow import StartEvent
from PIL import Image

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.executor import executor_agent
from tests.unit.device_fixtures import MemoryDevice


def test_executor_converts_path_screenshot_before_resizing(tmp_path, monkeypatch):
    path = tmp_path / "screen.png"
    Image.new("RGB", (2, 2)).save(path)
    state = MobileAgentState(screenshot=str(path))
    registry = SimpleNamespace(get_signatures=lambda **kwargs: {}, tools={})
    agent = executor_agent.ExecutorAgent(
        llm=None,
        registry=registry,
        action_ctx=SimpleNamespace(state_provider=object(), driver=MemoryDevice()),
        shared_state=state,
        agent_config=AgentConfig(),
        prompt_resolver=SimpleNamespace(get_prompt=lambda _: "prompt"),
    )
    agent.vision = True
    resize = AsyncMock(side_effect=lambda provider, image: image)
    monkeypatch.setattr(executor_agent, "_resize_model_screenshot_async", resize)
    ctx = SimpleNamespace(
        store=SimpleNamespace(set=AsyncMock()), write_event_to_stream=lambda event: None
    )
    asyncio.run(agent.prepare_context(ctx, StartEvent(subgoal="go")))
    assert resize.await_args.args[1].startswith(b"\x89PNG")

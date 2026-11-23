import asyncio
from types import SimpleNamespace

from llama_index.core.base.llms.types import TextBlock
from test_fast_agent_multi_block import (
    _assistant,
    _block,
    _click,
    _complete,
    _FakeLLM,
    _PromptResolver,
    _Registry,
    _StateProvider,
)

from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.fast_agent.fast_agent import FastAgent
from tests.unit.device_fixtures import MemoryDevice


class Provider:
    def __init__(self):
        self.seen = []

    async def get_context(self, context):
        self.seen.append(context)
        return f"tip {context.step}"


def test_fast_agent_context_each_step_and_none():
    async def run(provider):
        state = MobileAgentState(instruction="open the app", platform="android")
        llm = _FakeLLM(
            [
                _assistant(TextBlock(text=_block(_click(1)))),
                _assistant(TextBlock(text=_block(_complete("done")))),
            ]
        )
        config = SimpleNamespace(
            fast_agent=SimpleNamespace(vision=False, parallel_tools=True, tool_calling="auto"),
            max_steps=10,
            streaming=False,
            after_sleep_action=0,
        )
        agent = FastAgent(
            llm=llm,
            agent_config=config,
            registry=_Registry(state),
            action_ctx=SimpleNamespace(driver=MemoryDevice(), credentials=[], ui=None),
            state_provider=_StateProvider(),
            shared_state=state,
            prompt_resolver=_PromptResolver(),
            context_provider=provider,
        )
        await agent.run(input="open the app")
        return ["\n".join(message.content or "" for message in request) for request in llm.requests]

    provider = Provider()
    prompts = asyncio.run(run(provider))
    assert [item.step for item in provider.seen] == [1, 2]
    assert all(item.package_name == "com.example" for item in provider.seen)
    assert all(
        f"tip {i}" in prompt and "Additional context" in prompt
        for i, prompt in enumerate(prompts, 1)
    )
    assert all("Additional context" not in prompt for prompt in asyncio.run(run(None)))


def test_external_context_goal_and_none():
    from mobilerun.agents.workflows.external import run_with_context

    async def run(provider):
        async def external(**kwargs):
            return kwargs["instruction"]

        return await run_with_context(
            external,
            provider,
            MobileAgentState(instruction="goal"),
            device=object(),
            config={},
            max_steps=3,
        )

    provider = Provider()
    assert asyncio.run(run(provider)) == "goal\n\nAdditional context:\ntip 0"
    assert provider.seen[0].step == 0
    assert asyncio.run(run(None)) == "goal"


def test_mobile_agent_passes_context_to_external_goal(monkeypatch):
    from llama_index.core.workflow import StartEvent

    import mobilerun.agents.workflows.droid.droid_agent as module
    from mobilerun.agents.config.user_config import MobileConfig

    received = []

    async def external(**kwargs):
        received.append(kwargs["instruction"])
        return {"success": True, "reason": "done"}

    async def device(**kwargs):
        return object()

    async def noop(*args):
        pass

    monkeypatch.setattr(module, "load_agent", lambda name: {"run": external, "config": {}})
    monkeypatch.setattr(module.adb, "device", device)
    monkeypatch.setattr(module.MobileAgent, "_initialize_workflow_deadline", noop)
    monkeypatch.setattr(module.MobileAgent, "_initialize_run_trajectory", noop)
    config = MobileConfig()
    config.agent.name = "fake"
    config.device.serial = "fake-device"
    provider = Provider()
    agent = module.MobileAgent(goal="goal", config=config, context_provider=provider)
    ctx = SimpleNamespace(write_event_to_stream=lambda event: None)
    asyncio.run(agent.start_handler(ctx, StartEvent()))
    assert received == ["goal\n\nAdditional context:\ntip 0"]
    assert provider.seen[0].instruction == "goal"

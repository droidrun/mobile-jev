import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.workflow import StartEvent

from mobilerun.agents import AgentOperator
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.workflows.droid.droid_agent import MobileAgent
from tests.unit.device_fixtures import MemoryDevice


@pytest.mark.parametrize("mode", ["server", "composite"])
@pytest.mark.parametrize("server_url", [None, ""])
def test_app_card_missing_server_url_uses_local_backend(mode, server_url, tmp_path):
    from mobilerun.agents.context.providers.app_card_context_provider import (
        AppCardContextProvider,
    )

    folder = tmp_path / "android"
    folder.mkdir()
    (folder / "app_cards.json").write_text('{"com.example.app": "card.md"}')
    (folder / "card.md").write_text("LOCAL CARD")
    provider = AppCardContextProvider(mode=mode, server_url=server_url, app_cards_dir=str(tmp_path))

    assert (
        asyncio.run(provider.backend.fetch_card("com.example.app", platform="android"))
        == "LOCAL CARD"
    )


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_operator_custom_context_reaches_manager_each_step(monkeypatch, stateless, enabled):
    from mobilerun.agents import ContextProvider, StepContext

    seen = []

    class Tips(ContextProvider):
        async def get_context(self, context):
            seen.append(context)
            return "SETTINGS SEARCH TIP"

    config = MobileConfig()
    config.agent.reasoning = True
    config.agent.manager.stateless = stateless
    config.agent.app_cards.enabled = enabled
    config.agent.app_cards.mode = "unknown"
    provider = Tips()
    captured = []

    async def get_state():
        return SimpleNamespace(
            formatted_text="screen",
            focused_text="",
            elements=[{"text": "search"}],
            phone_state={
                "packageName": "com.android.settings",
                "currentApp": "Settings",
            },
        )

    async def run(agent):
        manager = agent.manager_agent
        assert manager.context_provider is provider
        manager.vision = False
        manager.save_trajectory = "none"
        manager.state_provider = SimpleNamespace(get_state=get_state)
        manager.action_ctx = SimpleNamespace(ui=None, driver=MemoryDevice())
        agent.shared_state.step_number = 3
        agent.shared_state.action_history = [{"name": "tap"}]
        agent.shared_state.summary_history = ["tapped"]
        agent.shared_state.action_outcomes = [True]
        agent.shared_state.error_descriptions = [""]
        for _ in range(2):
            await manager.prepare_context(
                SimpleNamespace(write_event_to_stream=lambda e: None), StartEvent()
            )
            captured.append(
                await (manager._build_prompt() if stateless else manager._build_system_prompt())
            )
        return SimpleNamespace(success=True, reason="done", structured_output=None, steps=3)

    monkeypatch.setattr(MobileAgent, "run", run)
    operator = AgentOperator(config=config, llms=MockLLM(), context_provider=provider)
    assert asyncio.run(operator.execute(object(), "open search")).success
    assert len(seen) == 2
    assert all(isinstance(item, StepContext) for item in seen)
    assert seen[0].package_name == "com.android.settings"
    assert seen[0].activity_name == "Settings"
    assert seen[0].instruction == "open search"
    assert seen[0].platform.lower() == "android"
    assert seen[0].step == 3
    assert seen[0].ui_elements == [{"text": "search"}]
    assert seen[0].last_actions[0]["summary"] == "tapped"
    assert seen[0].screenshot is None
    assert all("SETTINGS SEARCH TIP" in prompt for prompt in captured)


def test_app_card_context_provider_loads_bundled_gmail():
    from mobilerun.agents import AppCardContextProvider, StepContext

    provider = AppCardContextProvider()
    context = StepContext(
        package_name="com.google.android.gm",
        activity_name="",
        instruction="",
        platform="android",
        step=1,
        ui_elements=[],
        last_actions=[],
        screenshot=None,
    )
    assert "Gmail App Guide" in asyncio.run(provider.get_context(context))


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_no_custom_provider_respects_config(tmp_path, enabled, stateless):
    from mobilerun.agents.context.context_provider import update_context

    config = MobileConfig()
    config.agent.reasoning = True
    config.agent.manager.stateless = stateless
    config.agent.app_cards.enabled = enabled
    config.agent.app_cards.app_cards_dir = str(tmp_path)
    (tmp_path / "android").mkdir()
    (tmp_path / "android" / "app_cards.json").write_text('{"com.android.settings": "settings.md"}')
    (tmp_path / "android" / "settings.md").write_text("LOCAL CARD")
    agent = MobileAgent("open search", config=config, llms=MockLLM(), context_provider=None)
    agent.shared_state.current_package_name = "com.android.settings"
    agent.shared_state.platform = "android"
    asyncio.run(update_context(agent.manager_agent.context_provider, agent.shared_state, None))
    assert agent.shared_state.extra_context == ("LOCAL CARD" if enabled else "")


def test_context_screenshot_and_failure_clear_previous():
    from mobilerun.agents import ContextProvider
    from mobilerun.agents.context.context_provider import update_context
    from mobilerun.agents.workflows.droid.state import MobileAgentState

    class Provider(ContextProvider):
        async def get_context(self, context):
            assert context.screenshot == b"image"
            raise OSError("failure")

    state = MobileAgentState(extra_context="stale")
    asyncio.run(update_context(Provider(), state, "aW1hZ2U="))
    assert state.extra_context == ""

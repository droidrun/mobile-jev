import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.context.providers.local_provider import LocalCardBackend
from mobilerun.agents.tools.tool_registry import ToolRegistry
from mobilerun.agents.utils.prompt_resolver import PromptResolver
from mobilerun.agents.workflows.droid.state import MobileAgentState
from mobilerun.agents.workflows.executor.executor_agent import ExecutorAgent


class FakeStore:
    def __init__(self):
        self.values = {}

    async def set(self, key, value):
        self.values[key] = value

    async def get(self, key):
        return self.values[key]


class FakeWorkflowContext:
    def __init__(self):
        self.store = FakeStore()
        self.events = []

    def write_event_to_stream(self, event):
        self.events.append(event)


class AppCardsTest(unittest.TestCase):
    def test_executor_prompt_receives_shared_app_card(self):
        async def run():
            shared_state = MobileAgentState(
                instruction="Inspect the app safely",
                extra_context="APP_CARD_SENTINEL",
                formatted_device_state="Device state",
                plan="Plan text",
                progress_summary="Progress text",
                platform="Android",
            )
            agent = ExecutorAgent(
                llm=SimpleNamespace(),
                registry=ToolRegistry(),
                action_ctx=None,
                shared_state=shared_state,
                agent_config=AgentConfig(),
                prompt_resolver=PromptResolver(
                    {
                        "executor_system": (
                            "Instruction={{ instruction }}\n"
                            "AppCard={{ extra_context }}\n"
                            "State={{ device_state }}"
                        )
                    }
                ),
            )
            ctx = FakeWorkflowContext()

            await agent.prepare_context(
                ctx,
                SimpleNamespace(get=lambda key, default="": "Read visible UI"),
            )
            return ctx.store.values["executor_messages"][0].blocks[0].text

        prompt_text = asyncio.run(run())

        self.assertIn("APP_CARD_SENTINEL", prompt_text)

    def test_local_provider_loads_existing_app_card(self):
        async def run():
            provider = LocalCardBackend("assets/app_cards")
            return await provider.fetch_card("com.google.android.gm", "", "android")

        app_card = asyncio.run(run())

        self.assertIn("Gmail App Guide", app_card)
        self.assertIn("Search", app_card)
        self.assertIn("compose", app_card)

    def test_app_card_readme_uses_current_provider_api(self):
        readme = (
            Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
            / "assets/app_cards/README.md"
        ).read_text(encoding="utf-8")

        self.assertNotIn("AppCardLoader", readme)
        self.assertIn("AppCardContextProvider", readme)
        self.assertIn("await provider.get_context", readme)
        self.assertIn('platform="android"', readme)


if __name__ == "__main__":
    unittest.main()

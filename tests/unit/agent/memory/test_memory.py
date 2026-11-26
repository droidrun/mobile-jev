import asyncio
import json
import socket

import httpx
import pytest
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import InsertMethod
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem

from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock
from mobilerun_agents.agent.memory.blocks.retriever import RetrieverBlock
from mobilerun_agents.agent.memory.blocks.run_notes import RunNotesBlock
from mobilerun_agents.agent.memory.memory import build_memory, build_run_memory
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.config.user_config import MemoryConfig


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("memory tests cannot open network connections")

    for method in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, method, blocked)


@pytest.fixture
def snapshot():
    return StepSnapshot(
        goal="Find order",
        step=1,
        ui=UIState([], "", "", {"packageName": "mail", "activityName": "Inbox"}, 0, 0),
        device_context=[ContextItem(source="battery", kind="status", data={"level": 63})],
    )


def test_repeat_uses_the_call_id_when_two_actions_share_a_step(snapshot):
    from mobilerun_agents.agent.memory.blocks.repeat import RepeatBlock
    from mobilerun_agents.agent.memory.messages import action_messages
    from mobilerun_agents.agent.workflows.core.state import ActionRecord

    snapshot.ui.elements.append({"index": 3, "text": "first"})
    other = snapshot.model_copy(update={"ui": UIState([{"index": 4}], "", "", {}, 0, 0)})
    first = ActionRecord(step=1, name="tap", args={}, ok=True, summary="ok", before=snapshot)
    second = ActionRecord(step=1, name="tap", args={}, ok=True, summary="ok", before=other)
    snapshot._actions = (first, second)
    messages = action_messages(first, "first_call") + action_messages(second, "second_call")
    assert "Last action here: first_call" in asyncio.run(RepeatBlock().aget(messages, snapshot=snapshot))


class Docs(BaseRetriever):
    text = "inbox guide"

    def _retrieve(self, query_bundle):
        self.query = query_bundle.query_str
        return [NodeWithScore(node=TextNode(text=self.text))]


@pytest.mark.parametrize("enabled", [False, True])
def test_app_cards_are_opt_in(snapshot, enabled):
    snapshot.ui.phone_state["packageName"] = "com.google.android.gm"
    config = MemoryConfig(blocks=["app_cards"] if enabled else [])
    memory = build_run_memory(config, "android")
    messages = asyncio.run(memory.aget(snapshot=snapshot))
    assert any("<app_card>" in (message.content or "") for message in messages) == enabled


@pytest.mark.parametrize("method", list(InsertMethod))
def test_blocks_render_current_step_without_persisting_it(snapshot, method):
    docs = Docs()
    notes = RunNotesBlock()
    snapshot.notes.append("Order is pending")
    memory = build_memory(
        MemoryConfig(insert_method=method),
        [DeviceContextBlock(), RetrieverBlock(retriever=docs), notes],
    )

    async def check():
        await memory.aput(ChatMessage(role="user", content="hello"))
        result = await memory.aget(snapshot=snapshot)
        text = result[0].content
        assert result[0].role.value == method.value
        assert all(part in text for part in ["battery", "status", "63", "inbox guide"])
        assert text.count("Order is pending") == 1
        assert all(part in docs.query for part in ["Find order", "mail", "Inbox"])
        snapshot.notes.clear()
        assert "Order is pending" not in (await memory.aget(snapshot=snapshot))[0].content
        snapshot.device_context.clear()
        assert "battery" not in (await memory.aget(snapshot=snapshot))[0].content
        assert [m.content for m in await memory.aget_all()] == ["hello"]

    asyncio.run(check())


def test_budget_drops_retrieval_before_notes(snapshot):
    notes = RunNotesBlock()
    snapshot.notes.append("pending")
    docs = Docs()
    docs.text *= 100
    memory = build_memory(MemoryConfig(token_limit=16), [notes, RetrieverBlock(retriever=docs)])
    text = asyncio.run(memory.aget(snapshot=snapshot))[0].content
    assert "pending" in text and "inbox guide" not in text


def test_history_budget_keeps_newest_tool_turn():
    memory = build_memory(MemoryConfig(token_limit=100), [])

    async def check():
        for step in range(6):
            await memory.aput_messages(
                [
                    ChatMessage(role=role, content=f"{step} " + "word " * 8)
                    for role in ("user", "assistant", "tool", "assistant")
                ]
            )
        result = await memory.aget()
        assert result[0].role == "user" and not result[0].content.startswith("0 ")
        assert [m.role.value for m in result[-4:]] == ["user", "assistant", "tool", "assistant"]
        assert all(m.content.startswith("5 ") for m in result[-4:])

    asyncio.run(check())


def test_transient_server_failure_is_not_cached(snapshot):
    responses = iter([httpx.Response(503), httpx.Response(200, json={"app_card": "guide"})])

    async def check():
        async with httpx.AsyncClient(
            base_url="https://cards.test",
            transport=httpx.MockTransport(lambda request: next(responses)),
        ) as client:
            block = AppCardBlock(platform="ios", client=client)
            assert await block.aget(snapshot=snapshot) == ""
            assert await block.aget(snapshot=snapshot) == "guide"
            assert await block.aget(snapshot=snapshot) == "guide"

    asyncio.run(check())


def test_local_card_mapping_cannot_escape_platform_folder(snapshot, tmp_path):
    folder = tmp_path / "android"
    folder.mkdir()
    (tmp_path / "outside.md").write_text("private")
    (folder / "app_cards.json").write_text('{"mail":"../outside.md"}')
    block = AppCardBlock(platform="android", cards_dir=tmp_path)
    assert asyncio.run(block.aget(snapshot=snapshot)) == ""


def test_app_cards_server_then_platform_specific_local(snapshot, tmp_path):
    for platform in ["ios", "android"]:
        folder = tmp_path / platform
        folder.mkdir()
        (folder / "app_cards.json").write_text('{"mail":"mail.md"}')
        (folder / "mail.md").write_text(platform + " guide")

    def respond(request):
        assert request.url.path == "/app-cards"
        payload = json.loads(request.content)
        assert payload["platform"] == "ios" and payload["instruction"] == "Find order"
        if payload["package_name"] == "browser":
            return httpx.Response(200, json={"app_card": "server guide"})
        return httpx.Response(404)

    async def check():
        async with httpx.AsyncClient(
            base_url="https://cards.test", transport=httpx.MockTransport(respond)
        ) as client:
            block = AppCardBlock(platform="ios", cards_dir=tmp_path, client=client)
            assert await block.aget(snapshot=snapshot) == "ios guide"
            snapshot.ui.phone_state["packageName"] = "browser"
            assert await block.aget(snapshot=snapshot) == "server guide"
            snapshot.ui.phone_state.pop("packageName")
            assert await block.aget(snapshot=snapshot) == ""

    asyncio.run(check())

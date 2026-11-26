import asyncio
import socket
from unittest.mock import Mock

import pytest
from llama_index.core.base.llms.types import TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import InsertMethod
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from sqlalchemy.ext.asyncio import AsyncEngine

from mobilerun_agents.agent.memory import memory as memory_module
from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.agent.memory.memory import build_run_memory
from mobilerun_agents.agent.memory.messages import text_messages
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
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
        ui=UIState([], "", "", {"packageName": "mail", "activityName": "Inbox"}, 0, 0),
        device_context=[ContextItem(source="battery", kind="status", data={"level": 63})],
    )


def test_repeat_uses_the_call_id_when_two_actions_share_a_step(snapshot):
    snapshot.ui.elements.append({"index": 3, "text": "first"})
    other = snapshot.model_copy(update={"ui": UIState([{"index": 4}], "", "", {}, 0, 0)})
    calls = [
        ToolSelection(tool_id=str(i), tool_name="tap", tool_kwargs={"index": i}) for i in (3, 4)
    ]
    actions = [
        ActionRecord(step=1, call=call, ok=True, summary="ok", before=before)
        for call, before in zip(calls, (snapshot, other), strict=True)
    ]
    state = AgentState(current=snapshot, actions=actions)

    async def check():
        memory = build_run_memory(MemoryConfig(), "android")
        # Different displayed names distinguish each matched SQL-history call
        # from the state fallback; both ActionRecords share the same step.
        await memory.aput_messages(
            [
                ChatMessage(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(
                            tool_call_id=call.tool_id,
                            tool_name=f"wire_{call.tool_id}",
                            tool_kwargs=call.tool_kwargs,
                        )
                    ],
                )
                for call in calls
            ]
        )
        for current, expected, unexpected in [
            (snapshot, "wire_3", "wire_4"),
            (other, "wire_4", "wire_3"),
        ]:
            messages = await memory.aget(snapshot=current, state=state)
            hints = "\n".join(m.content or "" for m in messages)
            assert f"Last action here: {expected}" in hints
            assert f"Last action here: {unexpected}" not in hints

    asyncio.run(check())


@pytest.mark.parametrize("enabled", [False, True])
def test_app_cards_are_opt_in(snapshot, enabled):
    snapshot.ui.phone_state["packageName"] = "com.google.android.gm"
    config = MemoryConfig(blocks=["app_cards"] if enabled else [])
    memory = build_run_memory(config, "android")
    messages = asyncio.run(memory.aget(snapshot=snapshot))
    assert any("<app_card>" in (message.content or "") for message in messages) == enabled


@pytest.mark.parametrize("method", list(InsertMethod))
def test_blocks_render_current_step_without_persisting_it(snapshot, method):
    state = AgentState(goal="Find order", notes=["Order is pending"], current=snapshot)
    memory = build_run_memory(MemoryConfig(insert_method=method), "android")

    async def check():
        await memory.aput(ChatMessage(role="user", content="hello"))
        result = await memory.aget(snapshot=state.current, state=state)
        text = result[0].content
        assert result[0].role.value == method.value
        assert all(part in text for part in ["battery", "status", "63", "Find order"])
        assert text.count("Order is pending") == 1
        state.notes.clear()
        assert (
            "Order is pending"
            not in (await memory.aget(snapshot=state.current, state=state))[0].content
        )
        snapshot.device_context.clear()
        assert "battery" not in (await memory.aget(snapshot=snapshot))[0].content
        assert [m.content for m in await memory.aget_all()] == ["hello"]

    asyncio.run(check())


def test_default_memory_budget():
    memory = build_run_memory(MemoryConfig(), "android")
    assert (memory.token_limit, memory.chat_history_token_ratio, memory.token_flush_size) == (
        30000,
        0.8,
        2500,
    )


def test_local_card_mapping_cannot_escape_platform_folder(snapshot, tmp_path):
    folder = tmp_path / "android"
    folder.mkdir()
    (tmp_path / "outside.md").write_text("private")
    (folder / "app_cards.json").write_text('{"mail":"../outside.md"}')
    block = AppCardBlock(platform="android", cards_dir=tmp_path)
    assert asyncio.run(block.aget(snapshot=snapshot)) == ""


def test_local_card_uses_platform_and_caches_by_package(snapshot, tmp_path):
    for platform in ("ios", "android"):
        folder = tmp_path / platform
        folder.mkdir()
        (folder / "app_cards.json").write_text('{"mail":"mail.md"}')
        (folder / "mail.md").write_text(f"{platform} guide")

    block = AppCardBlock(platform="ios", cards_dir=tmp_path)

    async def check():
        state = AgentState(goal="Find order", current=snapshot)
        assert await block.aget(snapshot=snapshot, state=state) == "ios guide"
        (tmp_path / "ios" / "mail.md").write_text("updated guide")
        state.goal = "Find receipt"
        assert await block.aget(snapshot=snapshot, state=state) == "ios guide"
        snapshot.ui.phone_state["packageName"] = "browser"
        assert await block.aget(snapshot=snapshot, state=state) == ""
        snapshot.ui.phone_state.pop("packageName")
        assert await block.aget(snapshot=snapshot, state=state) == ""

    asyncio.run(check())


def test_text_projection_keeps_call_order_and_preserves_the_original_messages():
    user = ChatMessage(role="user", content="Action taken:")
    assistant = ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(tool_call_id="first", tool_name="tap", tool_kwargs={"index": 3}),
            TextBlock(text="Open the inbox"),
            ToolCallBlock(tool_call_id="second", tool_name="type", tool_kwargs={"text": "é"}),
        ],
    )
    result = ChatMessage(role="tool", content="Opened", additional_kwargs={"tool_call_id": "first"})

    projected = text_messages([user, assistant, result])

    assert [(message.role, message.content) for message in projected] == [
        ("user", "Action taken:"),
        ("assistant", 'Open the inbox\ntap({"index": 3})\ntype({"text": "é"})'),
        ("assistant", "Opened"),
    ]
    assert assistant.content == "Open the inbox"
    assert result.role == "tool"


def test_token_estimates_preserve_text_tool_arguments_and_message_metadata():
    memory = build_run_memory(MemoryConfig(), "android")
    memory.tokenizer_fn = list  # Count characters, including the literal é in tool arguments.
    text = TextBlock(text="hi")
    call = ToolCallBlock(tool_call_id="c", tool_name="tap", tool_kwargs={"x": "é"})
    message = ChatMessage(role="assistant", blocks=[text, call], additional_kwargs={"tag": "ok"})

    assert memory._estimate_token_count("hi") == 2
    assert memory._estimate_token_count(message) == 25
    assert memory._estimate_token_count([text, call]) == 12
    assert memory._estimate_token_count([message]) == 25


def test_close_disposes_the_runs_private_engine(monkeypatch):
    engine = Mock(spec=AsyncEngine)
    monkeypatch.setattr(memory_module, "_private_engine", lambda: engine)
    memory = build_run_memory(MemoryConfig(), "android")

    asyncio.run(memory.aclose())

    engine.dispose.assert_awaited_once_with()

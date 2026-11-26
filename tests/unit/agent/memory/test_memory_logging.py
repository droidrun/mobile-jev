import asyncio
import logging

from llama_index.core.llms import ChatMessage

from mobilerun_agents.agent.memory.memory import build_memory
from mobilerun_agents.agent.memory.messages import action_messages
from mobilerun_agents.agent.workflows.core.state import ActionOutcome
from mobilerun_agents.config.user_config import MemoryConfig


def test_memory_store_does_not_log_message_parameters(caplog):
    secret = "planted-sensitive-memory-value"

    async def read_and_write():
        memory = build_memory(MemoryConfig(), [])
        await memory.aput(ChatMessage(role="user", content=secret))
        assert (await memory.aget())[0].content == secret

    with caplog.at_level(logging.DEBUG, logger="aiosqlite"):
        with caplog.at_level(logging.DEBUG, logger="sqlalchemy.engine"):
            asyncio.run(read_and_write())

    assert all(secret not in record.getMessage() for record in caplog.records)


def test_large_tool_arguments_evict_the_oldest_complete_turn():
    async def read_and_write():
        memory = build_memory(MemoryConfig(token_limit=250), [])
        for step in range(1, 4):
            record = ActionOutcome(
                step=step, name="input_text", args={"text": " fake" * 85}, ok=True, summary="ok"
            )
            await memory.aput_messages(action_messages(record, "input_text"))
        messages = await memory.aget()
        assert [message.role for message in messages] == ["user", "assistant", "tool"]
        assert messages[-1].content.startswith("3 ·")

    asyncio.run(read_and_write())

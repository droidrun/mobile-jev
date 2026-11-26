import asyncio
import logging

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.memory.memory import build_run_memory
from mobilerun_agents.agent.memory.messages import remember
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord
from mobilerun_agents.config.user_config import MemoryConfig


def test_memory_sql_debug_never_logs_rows_or_arguments(caplog):
    secret = "planted-sensitive-memory-value"

    async def read_and_write():
        memory = build_run_memory(MemoryConfig(), "android")
        await memory.aput(ChatMessage(role="user", content=secret))
        assert (await memory.aget())[0].content == secret

    with caplog.at_level(logging.DEBUG, logger="aiosqlite"):
        with caplog.at_level(logging.DEBUG, logger="sqlalchemy.engine"):
            asyncio.run(read_and_write())
            logging.getLogger("aiosqlite").debug("unrelated host connection still logs")

    assert all(secret not in record.getMessage() for record in caplog.records)
    assert not any(record.name.startswith("sqlalchemy.engine") for record in caplog.records)
    assert "unrelated host connection still logs" in caplog.text


def test_200_token_budget_counts_arguments_and_evicts_a_complete_turn():
    async def read_and_write():
        snapshot = StepSnapshot(ui=UIState([], "", "", {}, 100, 200))
        memory = build_run_memory(MemoryConfig(token_limit=200), "android")
        for step in range(1, 4):
            call = ToolSelection(
                tool_id=f"call-{step}", tool_name="input_text", tool_kwargs={"text": " fake" * 85}
            )
            await remember(
                memory,
                call,
                ActionRecord(step=step, call=call, ok=True, summary="ok", before=snapshot),
            )
        messages = await memory.aget()
        assert [message.role for message in messages] == ["user", "assistant", "tool"]
        call = messages[1].blocks[0]
        assert isinstance(call, ToolCallBlock)
        assert call.tool_call_id == messages[2].additional_kwargs["tool_call_id"] == "call-3"
        assert messages[2].content.startswith("3 ·")

    asyncio.run(read_and_write())

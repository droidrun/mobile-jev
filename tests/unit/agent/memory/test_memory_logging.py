import asyncio
import logging

from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.memory.memory import build_executor_memory
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.config.user_config import MobileConfig


def test_memory_sql_debug_never_logs_rows_or_arguments(caplog):
    secret = "planted-sensitive-memory-value"

    async def read():
        memory = build_executor_memory(MobileConfig(), "android")
        try:
            assert secret in (await memory.aget(state=AgentState(goal=secret)))[0].content
        finally:
            await memory.aclose()

    with caplog.at_level(logging.DEBUG, logger="aiosqlite"):
        with caplog.at_level(logging.DEBUG, logger="sqlalchemy.engine"):
            asyncio.run(read())
            logging.getLogger("aiosqlite").debug("unrelated host connection still logs")

    assert all(secret not in record.getMessage() for record in caplog.records)
    assert not any(record.name.startswith("sqlalchemy.engine") for record in caplog.records)
    assert "unrelated host connection still logs" in caplog.text


def test_token_budget_counts_last_step_arguments_and_truncates_history():
    async def read():
        snapshot = StepSnapshot(ui=UIState([], "", "", {}, 100, 200))
        memory = build_executor_memory(MobileConfig(memory={"token_limit": 100}), "android")
        state = AgentState(current=snapshot)
        try:
            for step in range(1, 4):
                call = ToolSelection(
                    tool_id=f"call-{step}",
                    tool_name="input_text",
                    tool_kwargs={"text": " fake" * 85},
                )
                state.actions.append(
                    ActionRecord(step=step, call=call, ok=True, summary="ok", before=snapshot)
                )
            (message,) = await memory.aget(state=state)
            assert "Step 3" in message.content and " fake" * 85 in message.content
            assert "<history>" not in message.content
            assert await memory.aget_all() == []
        finally:
            await memory.aclose()

    asyncio.run(read())

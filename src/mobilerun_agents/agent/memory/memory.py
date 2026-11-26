"""Create the private, budgeted memory for one workflow invocation."""

import json
import logging
import sqlite3
from collections.abc import Sequence
from threading import current_thread
from typing import Any, Literal, cast

from llama_index.core.base.llms.types import ChatMessage, ContentBlock, ToolCallBlock
from llama_index.core.memory import BaseMemoryBlock, Memory
from pydantic import PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock
from mobilerun_agents.agent.memory.blocks.notes import NotesBlock
from mobilerun_agents.agent.memory.blocks.repeat import RepeatBlock
from mobilerun_agents.config.paths import resolve_prompt_path
from mobilerun_agents.config.user_config import AgentConfig, MemoryConfig


class _MemoryConnection(sqlite3.Connection):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # aiosqlite logs bound parameters from its dedicated connection thread.
        vars(current_thread())["_mobilerun_memory_connection"] = True


def _allow_driver_log(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING or not getattr(
        current_thread(), "_mobilerun_memory_connection", False
    )


def _private_engine() -> AsyncEngine:
    logging.getLogger("aiosqlite").addFilter(_allow_driver_log)
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        hide_parameters=True,
        connect_args={"factory": _MemoryConnection},
    )
    # SQLAlchemy DEBUG includes result rows, even with hide_parameters=True.
    # An unregistered logger cannot inherit the host's log level or handlers.
    logger = logging.Logger("mobilerun.memory.sql", level=logging.WARNING)
    logger.propagate = False
    engine.sync_engine.logger = logger
    return engine


class _RunMemory(Memory):
    _engine: AsyncEngine = PrivateAttr()

    async def aclose(self) -> None:
        """Release the private SQLite connection after the workflow stops using memory."""
        await self._engine.dispose()

    def _estimate_token_count(
        self, message_or_blocks: str | ChatMessage | list[ChatMessage] | list[ContentBlock]
    ) -> int:
        # The upstream estimator ignores calls in messages and rejects calls in
        # a bare block list. Only correct those omissions; delegate all else.
        items: Sequence[ChatMessage | ContentBlock]
        match message_or_blocks:
            case str():
                return super()._estimate_token_count(message_or_blocks)
            case ChatMessage():
                items = [message_or_blocks]
            case list():
                items = message_or_blocks

        extra = 0
        filtered: list[ChatMessage | ContentBlock] = []
        for item in items:
            match item:
                case ChatMessage(blocks=blocks):
                    filtered.append(item)
                case ToolCallBlock():
                    blocks = [item]
                case _:
                    filtered.append(item)
                    continue
            for block in blocks:
                match block:
                    case ToolCallBlock(tool_kwargs=kwargs):
                        extra += len(self.tokenizer_fn(json.dumps(kwargs, ensure_ascii=False)))
        if len(filtered) != len(items):
            return super()._estimate_token_count(cast(list[ContentBlock], filtered)) + extra
        return super()._estimate_token_count(message_or_blocks) + extra


def build_run_memory(
    config: MemoryConfig, platform: str, agent: AgentConfig | None = None
) -> _RunMemory:
    blocks: list[BaseMemoryBlock[str]] = [
        DeviceContextBlock(),
        NotesBlock(),
        RepeatBlock(config=agent or AgentConfig()),
    ]
    if "app_cards" in config.blocks and platform in ("android", "ios"):
        blocks.append(
            AppCardBlock(
                platform=cast(Literal["android", "ios"], platform),
                cards_dir=resolve_prompt_path("assets/app_cards"),
            )
        )
    engine = _private_engine()
    memory = cast(
        _RunMemory,
        _RunMemory.from_defaults(
            token_limit=config.token_limit,
            chat_history_token_ratio=0.8,
            token_flush_size=2500,
            insert_method=config.insert_method,
            memory_blocks=blocks,
            async_engine=engine,
        ),
    )
    memory._engine = engine
    return memory

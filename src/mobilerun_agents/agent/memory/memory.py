"""Create the private, budgeted memory for one workflow invocation."""

import logging
import sqlite3
from collections.abc import Callable, Mapping
from threading import current_thread
from typing import Any, cast

from llama_index.core.base.llms.types import ChatMessage, ContentBlock, TextBlock
from llama_index.core.memory import BaseMemoryBlock, Memory
from llama_index.core.prompts import RichPromptTemplate
from pydantic import PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from workflows import Context

from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState
from mobilerun_agents.config.user_config import MemoryConfig, MobileConfig

from .registry import NO_MEMORY_BLOCKS, BlockContext, BlockFactory, configured_blocks


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
    _owned_engine: AsyncEngine | None = PrivateAttr(default=None)

    async def aclose(self) -> None:
        """Release the private SQLite connection after the workflow stops using memory."""
        if self._owned_engine is not None:
            await self._owned_engine.dispose()


class _BlockPrompt(RichPromptTemplate):
    def __init__(self) -> None:
        super().__init__(template_str="", template_vars=["memory_blocks"])

    def format_messages(self, llm: Any = None, **kwargs: Any) -> list[ChatMessage]:
        # The default template eagerly opens path images through Banks, whose
        # media root doesn't include the run's configurable screenshot folder.
        lines = ["<memory>"]
        media: list[ContentBlock] = []
        for name, blocks in kwargs["memory_blocks"]:
            lines.append(f"<{name}>")
            for block in blocks:
                match block:
                    case TextBlock(text=text):
                        lines.append(text)
                    case _:
                        media.append(block)
            lines.append(f"</{name}>")
        lines.append("</memory>")
        return [ChatMessage(role="user", blocks=[TextBlock(text="\n".join(lines)), *media])]


async def read_memory(
    memory: Memory, ctx: Context[AgentState] | Context[PlanningState]
) -> list[ChatMessage]:
    """Render memory from the latest persisted workflow state."""
    state = await ctx.store.get_state()
    return await memory.aget(state=state)


def build_executor_memory(
    config: MobileConfig,
    platform: str,
    async_engine: AsyncEngine | None = None,
    session_id: str | None = None,
    extra: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS,
) -> _RunMemory:
    blocks = configured_blocks(BlockContext("executor", platform, {}, config), extra)
    return _build_memory(config.memory, blocks, async_engine, session_id)


def build_planner_memory(
    config: MobileConfig,
    state: PlanningState,
    platform: str,
    extra: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS,
) -> _RunMemory:
    blocks = configured_blocks(BlockContext("planner", platform, {}, config, state), extra)
    return _build_memory(config.memory, blocks)


def prepare_child_memory(
    config: MobileConfig,
    platform: str,
    extra: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS,
) -> Callable[[], _RunMemory]:
    """Build the first child's blocks up front, without opening a memory engine."""
    first: list[BaseMemoryBlock[Any]] | None = configured_blocks(
        BlockContext("executor", platform, {}, config), extra
    )

    def new_memory() -> _RunMemory:
        nonlocal first
        if first is None:
            return build_executor_memory(config, platform, extra=extra)
        blocks, first = first, None
        return _build_memory(config.memory, blocks)

    return new_memory


def _build_memory(
    config: MemoryConfig,
    blocks: list[BaseMemoryBlock[Any]],
    async_engine: AsyncEngine | None = None,
    session_id: str | None = None,
) -> _RunMemory:
    owned_engine = _private_engine() if async_engine is None else None
    memory = cast(
        _RunMemory,
        _RunMemory.from_defaults(
            session_id=session_id,
            token_limit=config.token_limit,
            chat_history_token_ratio=0,
            insert_method=config.insert_method,
            memory_blocks=blocks,
            memory_blocks_template=_BlockPrompt(),
            async_engine=async_engine if async_engine is not None else owned_engine,
        ),
    )
    memory._owned_engine = owned_engine
    return memory

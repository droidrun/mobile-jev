"""Create the private, budgeted memory for one workflow invocation."""

import logging
import sqlite3
from threading import current_thread
from typing import Any, Literal, cast

from llama_index.core.base.llms.types import ChatMessage, ContentBlock, TextBlock
from llama_index.core.memory import BaseMemoryBlock, Memory, StaticMemoryBlock
from llama_index.core.prompts import RichPromptTemplate
from pydantic import PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from workflows import Context

from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.agent.memory.blocks.findings import FindingsBlock
from mobilerun_agents.agent.memory.blocks.goal import GoalBlock
from mobilerun_agents.agent.memory.blocks.history import HistoryBlock
from mobilerun_agents.agent.memory.blocks.last_step import LastStepBlock
from mobilerun_agents.agent.memory.blocks.subgoals import LastSubgoalBlock, SubgoalHistoryBlock
from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState
from mobilerun_agents.config.paths import resolve_prompt_path
from mobilerun_agents.config.prompts import render_prompt
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
    memory: Memory, ctx: Context[AgentState] | Context[PlanningState], redact: Redactor
) -> list[ChatMessage]:
    """Render memory from the latest persisted workflow state."""
    state = await ctx.store.get_state()
    return await memory.aget(state=state, redact=redact)


def build_setup_block(text: str) -> StaticMemoryBlock:
    """Build the pinned setup block for the executor and planner memory builders."""
    return StaticMemoryBlock(name="setup", static_content=[TextBlock(text=text)], priority=0)


def build_executor_memory(
    config: MemoryConfig,
    platform: str,
    agent: AgentConfig | None = None,
    async_engine: AsyncEngine | None = None,
    session_id: str | None = None,
) -> _RunMemory:
    blocks: list[BaseMemoryBlock[Any]] = []
    for name in config.executor.blocks:
        match name:
            case "setup":
                blocks.append(
                    build_setup_block(render_prompt("assets/prompts/memory/setup_executor.jinja2"))
                )
            case "goal":
                blocks.append(GoalBlock())
            case "last_step":
                blocks.append(LastStepBlock(screenshot=config.executor.last_step_screenshot))
            case "history":
                blocks.append(
                    HistoryBlock(steps=config.executor.history, config=agent or AgentConfig())
                )
            case "app_cards" if platform in ("android", "ios"):
                blocks.append(
                    AppCardBlock(
                        platform=cast(Literal["android", "ios"], platform),
                        cards_dir=resolve_prompt_path("assets/app_cards"),
                    )
                )
    return _build_memory(config, blocks, async_engine, session_id)


def build_planner_memory(config: MemoryConfig, state: PlanningState) -> _RunMemory:
    blocks: list[BaseMemoryBlock[Any]] = []
    for name in config.planner.blocks:
        match name:
            case "setup":
                blocks.append(
                    build_setup_block(
                        render_prompt(
                            "assets/prompts/memory/setup_planner.jinja2", executor=state.executor
                        )
                    )
                )
            case "goal":
                blocks.append(GoalBlock(include_notes=False))
            case "findings":
                blocks.append(FindingsBlock())
            case "last_subgoal":
                blocks.append(LastSubgoalBlock())
            case "subgoal_history":
                blocks.append(SubgoalHistoryBlock(history=config.planner.history))
    return _build_memory(config, blocks)


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

"""Run-scoped, token-budgeted history and per-observation context."""

import json
import logging
from abc import abstractmethod
from typing import Any, Literal, cast

from llama_index.core.base.llms.types import ChatMessage, ContentBlock, ToolCallBlock
from llama_index.core.memory import BaseMemoryBlock, Memory
from sqlalchemy.ext.asyncio import create_async_engine

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.config.paths import resolve_prompt_path
from mobilerun_agents.config.user_config import MemoryConfig

_ENGINE_LOG_NAME = "mobilerun_agent_memory"
# SQL result rows also contain raw memory. Keep this engine out of host debug logs.
_engine_logger = logging.getLogger(f"sqlalchemy.engine.Engine.{_ENGINE_LOG_NAME}")
_engine_logger.setLevel(logging.WARNING)
_engine_logger.propagate = False
logging.getLogger("aiosqlite").addFilter(lambda record: record.levelno > logging.DEBUG)


class RunMemory(Memory):
    def _estimate_token_count(
        self, message_or_blocks: str | ChatMessage | list[ChatMessage] | list[ContentBlock]
    ) -> int:
        if isinstance(message_or_blocks, ChatMessage):
            blocks = message_or_blocks.blocks
        elif isinstance(message_or_blocks, list):
            blocks = [
                block for item in message_or_blocks
                for block in (item.blocks if isinstance(item, ChatMessage) else [item])
            ]
        else:
            blocks = []
        extra = sum(
            len(self.tokenizer_fn(json.dumps(block.tool_kwargs, ensure_ascii=False)))
            for block in blocks if isinstance(block, ToolCallBlock)
        )
        if isinstance(message_or_blocks, list) and any(isinstance(b, ToolCallBlock) for b in message_or_blocks):
            message_or_blocks = [b for b in cast(list[ContentBlock], message_or_blocks) if not isinstance(b, ToolCallBlock)]
        return super()._estimate_token_count(message_or_blocks) + extra


def build_memory(config: MemoryConfig, blocks: list[BaseMemoryBlock[Any]]) -> Memory:
    """Build once per run with fresh blocks; Memory manages chat turn eviction."""
    return RunMemory.from_defaults(
        token_limit=config.token_limit,
        insert_method=config.insert_method,
        memory_blocks=blocks,
        chat_history_token_ratio=0.8,
        token_flush_size=max(1, config.token_limit // 12),
        async_engine=create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            hide_parameters=True,
            logging_name=_ENGINE_LOG_NAME,
        ),
    )


class SnapshotBlock(BaseMemoryBlock[str]):
    """Observations are read on demand, never learned from archived chat."""

    accept_short_term_memory: bool = False

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        snapshot: StepSnapshot | None = None,
        **kwargs: object,
    ) -> str:
        if snapshot is None:
            raise ValueError(f"{self.name} requires memory.aget(snapshot=...)")
        return await self.render(snapshot)

    async def _aput(self, messages: list[ChatMessage]) -> None:
        pass

    @abstractmethod
    async def render(self, snapshot: StepSnapshot) -> str: ...


def build_run_memory(config: MemoryConfig, platform: str) -> Memory:
    from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
    from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock

    blocks: list[BaseMemoryBlock[Any]] = [DeviceContextBlock()]
    if "app_cards" in config.blocks and (platform == "android" or platform == "ios"):
        blocks.append(
            AppCardBlock(
                platform=cast(Literal["android", "ios"], platform),
                cards_dir=resolve_prompt_path("assets/app_cards"),
            )
        )
    return build_memory(config, blocks)

"""Run-scoped, token-budgeted history and per-observation context."""

from abc import abstractmethod
from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemoryBlock, Memory

from mobilerun_agents.agent.memory.snapshot import StepSnapshot
from mobilerun_agents.config.user_config import MemoryConfig


def build_memory(config: MemoryConfig, blocks: list[BaseMemoryBlock[Any]]) -> Memory:
    """Build once per run with fresh blocks; Memory manages chat turn eviction."""
    return Memory.from_defaults(
        token_limit=config.token_limit,
        insert_method=config.insert_method,
        memory_blocks=blocks,
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

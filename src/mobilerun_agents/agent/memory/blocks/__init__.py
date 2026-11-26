"""Run-scoped memory blocks for mobile observations."""

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemoryBlock


class ObservationBlock(BaseMemoryBlock[str]):
    """Read live state, not chat history that has already been archived."""

    accept_short_term_memory: bool = False

    async def _aput(self, messages: list[ChatMessage]) -> None:
        pass

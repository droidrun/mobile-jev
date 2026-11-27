"""Run-scoped memory blocks that render the persisted workflow state."""

from abc import abstractmethod
from typing import Generic, TypeVar

from llama_index.core.base.llms.types import ChatMessage, ContentBlock
from llama_index.core.memory import BaseMemoryBlock
from pydantic import BaseModel

StateT = TypeVar("StateT", bound=BaseModel)
ContentT = TypeVar("ContentT", str, list[ContentBlock])


class StateBlock(BaseMemoryBlock[ContentT], Generic[StateT, ContentT]):
    """Render context from the workflow state, ignoring short-term memory."""

    accept_short_term_memory: bool = False

    async def _aput(self, messages: list[ChatMessage]) -> None:
        pass

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        state: StateT | None = None,
        **kwargs: object,
    ) -> ContentT:
        if state is None:
            raise TypeError(f"{type(self).__name__} requires the workflow state (state=...)")
        return await self.render(state)

    @abstractmethod
    async def render(self, state: StateT) -> ContentT:
        """Return the context to include for this state."""

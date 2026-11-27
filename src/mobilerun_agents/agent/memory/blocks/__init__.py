"""Run-scoped memory blocks that render the persisted workflow state."""

from abc import abstractmethod
from typing import Generic, TypeVar

from llama_index.core.base.llms.types import ChatMessage, ContentBlock
from llama_index.core.memory import BaseMemoryBlock
from pydantic import BaseModel

from mobilerun_agents.agent.logs.redaction import Redactor

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
        redact: Redactor | None = None,
        **kwargs: object,
    ) -> ContentT:
        if state is None:
            raise TypeError(f"{type(self).__name__} requires the workflow state (state=...)")
        if redact is None:
            raise TypeError(f"{type(self).__name__} requires the run redactor (redact=...)")
        return await self.render(state, redact)

    @abstractmethod
    async def render(self, state: StateT, redact: Redactor) -> ContentT:
        """Return the context to include for this state."""


def clip(text: str, limit: int, redact: Redactor) -> str:
    if 0 < limit < len(text):
        for start, end in redact.masked_spans(text):
            if start < limit < end:
                return text[:start]
    return text[:limit]

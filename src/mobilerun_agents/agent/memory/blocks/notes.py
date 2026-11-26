from llama_index.core.base.llms.types import ChatMessage

from mobilerun_agents.agent.workflows.core.state import AgentState

from . import ObservationBlock


class NotesBlock(ObservationBlock):
    """Render the current goal and notes without storing either in chat."""

    name: str = "notes"
    priority: int = 0

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        state: AgentState | None = None,
        **kwargs: object,
    ) -> str:
        return "\n".join([f"Goal: {state.goal}", *state.notes]) if state is not None else ""

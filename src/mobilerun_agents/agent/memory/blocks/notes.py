from mobilerun.core.devices import Device
from mobilerun.core.extensions.registry import redact_action_value
from workflows import Context

from mobilerun_agents.agent.tools.base import Tool, tool
from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock


@tool
async def note(text: str, ctx: Context[AgentState | PlanningState], device: Device) -> str:
    """Remember a useful fact for later turns. Do not include credentials or codes."""
    text = text.strip()
    if not text:
        return "No note saved: text is empty."
    redact_code = device.known_codes_redactor()
    safe = str(redact_action_value(text, contains_code=lambda value: redact_code(value) != value))
    async with ctx.store.edit_state() as state:
        if safe in state.notes:
            return "Note already saved."
        state.notes.append(safe)
    return "Note saved."


class NotesBlock(StateBlock[AgentState | PlanningState, str]):
    name: str = "notes"
    priority: int = 0

    def tools(self) -> tuple[Tool, ...]:
        return (note,)

    async def render(self, state: AgentState | PlanningState) -> str:
        return (
            render_prompt("assets/prompts/memory/notes.jinja2", notes=state.notes)
            if state.notes
            else ""
        )

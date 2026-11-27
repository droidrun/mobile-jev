from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock


class FindingsBlock(StateBlock[PlanningState, str]):
    name: str = "findings"
    priority: int = 0

    async def render(self, state: PlanningState, redact: Redactor) -> str:
        return (
            render_prompt("assets/prompts/memory/findings.jinja2", notes=state.notes)
            if state.notes
            else ""
        )

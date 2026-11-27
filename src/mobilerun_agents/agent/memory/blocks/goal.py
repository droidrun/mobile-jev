from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock


class GoalBlock(StateBlock[AgentState | PlanningState, str]):
    """Render the current goal and notes without storing either in chat."""

    name: str = "goal"
    priority: int = 0
    include_notes: bool = True

    async def render(self, state: AgentState | PlanningState) -> str:
        return render_prompt(
            "assets/prompts/memory/goal.jinja2",
            goal=state.goal,
            notes=state.notes if self.include_notes else (),
        )

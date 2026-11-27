import json

from llama_index.core.base.llms.types import ContentBlock, TextBlock

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.agent.workflows.core.targets import element_label
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock


def _screen(snapshot: StepSnapshot | None) -> dict[str, object]:
    phone = snapshot.ui.phone_state if snapshot is not None else {}
    focus = phone.get("focusedElement")
    match focus:
        case dict() as element if element:
            focus = element_label(
                element,
                limit=None,
                focus=element if phone.get("isEditable") else None,
            )
    return {
        "package": phone.get("packageName"),
        "activity": phone.get("activityName"),
        "focus": focus or None,
        "keyboard": phone.get("keyboardVisible"),
    }


class LastStepBlock(StateBlock[AgentState, list[ContentBlock]]):
    name: str = "last_step"
    priority: int = 0
    screenshot: bool = True

    async def render(self, state: AgentState) -> list[ContentBlock]:
        if not state.actions:
            return []
        step = max(record.step for record in state.actions)
        records = [record for record in state.actions if record.step == step]
        first = records[0]
        before = first.before
        screenshot = before.screenshot if self.screenshot else None
        text = render_prompt(
            "assets/prompts/memory/last_step.jinja2",
            step=step,
            calls=[
                (
                    record.call.tool_name,
                    json.dumps(record.call.tool_kwargs),
                    record.ok,
                    record.summary,
                )
                for record in records
            ],
            thought=first.thought,
            before=_screen(before),
            after=_screen(state.current),
            screen_changed=first.screen_changed,
            has_screenshot=screenshot is not None,
        )
        result: list[ContentBlock] = [TextBlock(text=text)]
        if screenshot is not None:
            result.append(screenshot)
        return result

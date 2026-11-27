import json

from llama_index.core.base.llms.types import ContentBlock, ImageBlock, TextBlock

from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.agent.workflows.core.targets import element_label
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock


def _screen(snapshot: StepSnapshot | None, redact: Redactor) -> dict[str, object]:
    phone = snapshot.ui.phone_state if snapshot is not None else {}
    focus = phone.get("focusedElement")
    match focus:
        case dict() as element if element:
            focus = element_label(
                element,
                limit=None,
                mask=redact.text,
                focus=element if phone.get("isEditable") else None,
            )
    return {
        "package": redact.value(phone.get("packageName")),
        "activity": redact.value(phone.get("activityName")),
        "focus": redact.value(focus or None),
        "keyboard": phone.get("keyboardVisible"),
    }


class LastStepBlock(StateBlock[AgentState, list[ContentBlock]]):
    name: str = "last_step"
    priority: int = 0
    screenshot: bool = True

    async def render(self, state: AgentState, redact: Redactor) -> list[ContentBlock]:
        if not state.actions:
            return []
        step = max(record.step for record in state.actions)
        records = [record for record in state.actions if record.step == step]
        first = records[0]
        before = first.before
        path = before.screenshot.path if self.screenshot and before.screenshot is not None else None
        text = render_prompt(
            "assets/prompts/memory/last_step.jinja2",
            step=step,
            calls=[
                (
                    record.call.tool_name,
                    json.dumps(redact.value(record.call.tool_kwargs)),
                    record.ok,
                    # Keep the latest result raw so the executor can use a just-fetched code.
                    record.summary,
                )
                for record in records
            ],
            thought=redact.text(first.thought) if first.thought else None,
            before=_screen(before, redact),
            after=_screen(state.current, redact),
            screen_changed=first.screen_changed,
            has_screenshot=path is not None,
        )
        result: list[ContentBlock] = [TextBlock(text=text)]
        if path is not None:
            result.append(ImageBlock(path=path))
        return result

from itertools import chain

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
from pydantic import Field

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.agent.workflows.core.targets import (
    cycle_visits,
    repeated_target,
    target,
    unchanged_steps,
)
from mobilerun_agents.config.user_config import AgentConfig

from . import ObservationBlock


class RepeatBlock(ObservationBlock):
    name: str = "repeat"
    priority: int = 0
    config: AgentConfig = Field(default_factory=AgentConfig)

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        snapshot: StepSnapshot | None = None,
        state: AgentState | None = None,
        **kwargs: object,
    ) -> str:
        if state is None or not state.actions:
            return ""
        lines = []
        latest = state.actions[-1]
        repeats = repeated_target(state.actions)
        if repeats >= self.config.repeat_warn_after:
            # Keep full strings so output consumers can mask secrets before any shortening.
            lines.append(
                f"{latest.call.tool_name} on {target(latest, limit=None)} repeated {repeats}×"
                " — it isn't working; try a different element or action"
            )
        fingerprint = snapshot.fingerprint if snapshot is not None else None
        if fingerprint is not None:
            last = next(
                (
                    record
                    for record in reversed(state.actions)
                    if record.before.fingerprint == fingerprint
                ),
                None,
            )
            if last is not None:
                lines.append(_last_action_note(last, messages))
        if (unchanged := unchanged_steps(state.actions)) >= self.config.no_progress_note_after:
            lines.append(
                f"The screen was unchanged after {unchanged} consecutive steps. Try a different approach or complete."
            )
        elif (visits := cycle_visits(state.actions, fingerprint)) >= self.config.cycle_note_after:
            lines.append(
                f"You have been on this screen {visits} times. Try a different approach or complete."
            )
        return "\n".join(lines)


def _last_action_note(last: ActionRecord, messages: list[ChatMessage] | None) -> str:
    name = last.call.tool_name
    for block in chain.from_iterable(assistant.blocks for assistant in reversed(messages or [])):
        match block:
            case ToolCallBlock(tool_call_id=tool_id, tool_name=tool_name) if (
                tool_id == last.call.tool_id
            ):
                name = tool_name
                break
    label = target(last, limit=None)
    action = f"{name} on {label}" if label else name
    status = "✓" if last.ok else "✗"
    change = {True: "screen changed", False: "screen unchanged", None: "screen change unknown"}
    return (
        f"You were on this screen at step {last.step}. Last action here: "
        f"{action} {status} · {change[last.screen_changed]}."
    )

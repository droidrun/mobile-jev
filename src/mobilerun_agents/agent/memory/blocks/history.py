import json

from llama_index.core.utils import get_tokenizer
from pydantic import Field

from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.agent.workflows.core.targets import (
    cycle_visits,
    repeated_target,
    target,
    unchanged_steps,
)
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig

from . import StateBlock

_CHANGE = {True: "screen changed", False: "screen unchanged", None: "screen change unknown"}


class HistoryBlock(StateBlock[AgentState, str]):
    name: str = "history"
    priority: int = 2
    steps: int = 5
    config: AgentConfig = Field(default_factory=AgentConfig)

    async def render(self, state: AgentState) -> str:
        if not state.actions:
            return ""
        by_step: dict[int, list[ActionRecord]] = {}
        for record in state.actions:
            by_step.setdefault(record.step, []).append(record)
        steps = [
            (
                step,
                [_action(record) for record in by_step[step]],
                _CHANGE[by_step[step][0].screen_changed],
            )
            for step in sorted(by_step)[-self.steps - 1 : -1]
        ]
        return render_prompt(
            "assets/prompts/memory/history.jinja2",
            steps=steps,
            warnings=self._warnings(state),
        )

    async def atruncate(self, content: str | list[object], tokens_to_truncate: int) -> str:
        if not content:
            return ""
        # Upstream supplies [] if the block was absent from the rendered content.
        assert type(content) is str
        tokenize = get_tokenizer()
        lines = content.splitlines()
        budget = len(tokenize(content)) - tokens_to_truncate
        for start in range(len(lines)):
            remaining = "\n".join(lines[start:])
            if len(tokenize(remaining)) <= budget:
                return remaining
        return ""

    def _warnings(self, state: AgentState) -> list[str]:
        lines: list[str] = []
        latest = state.actions[-1]
        repeats = repeated_target(state.actions)
        if repeats >= self.config.repeat_warn_after:
            lines.append(
                f"{latest.call.tool_name} on {target(latest, limit=None)} repeated {repeats}×"
                " — it isn't working; try a different element or action"
            )
        fingerprint = state.current.fingerprint if state.current is not None else None
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
                lines.append(_last_action_note(last))
        if (unchanged := unchanged_steps(state.actions)) >= self.config.no_progress_note_after:
            lines.append(
                f"The screen was unchanged after {unchanged} consecutive steps. Try a different approach or complete."
            )
        elif (visits := cycle_visits(state.actions, fingerprint)) >= self.config.cycle_note_after:
            lines.append(
                f"You have been on this screen {visits} times. Try a different approach or complete."
            )
        return lines


def _action(record: ActionRecord) -> str:
    index = record.call.tool_kwargs.get("index")
    indexed = f" [{index}]" if type(index) is int else ""
    args = {
        key: value
        for key, value in record.call.tool_kwargs.items()
        if key != "index" or not indexed
    }
    parameters = json.dumps(args, separators=(",", ":")) if args else ""
    label = target(record, limit=None)
    status = "✓" if record.ok else "✗"
    summary = " ".join(record.summary[:120].split())
    return (
        f"{record.call.tool_name}{indexed}{' ' + parameters if parameters else ''}"
        f"{' ' + label if label else ''} {status}{': ' + summary if summary else ''}"
    )


def _last_action_note(last: ActionRecord) -> str:
    name = last.call.tool_name
    label = target(last, limit=None)
    action = f"{name} on {label}" if label else name
    status = "✓" if last.ok else "✗"
    return (
        f"You were on this screen at step {last.step}. Last action here: "
        f"{action} {status} · {_CHANGE[last.screen_changed]}."
    )

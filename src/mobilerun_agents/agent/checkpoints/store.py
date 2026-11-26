"""Publish a compact state checkpoint after its observation files are on disk."""

import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from mobilerun.macro.models import build_screen_snapshot
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.logs.redaction import (
    redact_run_arguments,
    redact_run_text,
    run_sensitive_values,
)
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS

from .macro_export import macro_data, validate_vault_reference


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(data)
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def _json(data: object) -> bytes:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


class RunRecorder:
    def __init__(self, folder: Path, macro_path: Path, max_steps: int = DEFAULT_MAX_STEPS) -> None:
        self.folder = folder
        self.macro_path = macro_path
        self._state = AgentState(max_steps=max_steps)
        self._pre_states: list[dict[str, JSONValue]] = []
        self._decision_count = 0

    async def on_decision(self, choices: list[dict[str, JSONValue]]) -> str:
        """Write one redacted option list per Jev call, including retries."""
        self._decision_count += 1
        path = f"decisions/{self._decision_count:04}.json"
        payload = {"choices": [redact_run_arguments(choice) for choice in choices]}
        await asyncio.to_thread(_atomic, self.folder / path, _json(payload))
        return path

    async def on_step(self, record: ActionRecord) -> None:
        validate_vault_reference(record)
        state = self._state
        if state.step != record.step:
            self._pre_states.clear()
        tree_path = f"ui_states/{record.step:04}.json"
        guard = redact_run_arguments(build_screen_snapshot(record.before.ui))
        trees = [*self._pre_states, guard]
        action = record.model_copy(
            update={
                "name": redact_run_text(record.name),
                "args": {},
                "summary": "[REDACTED]",
                "thought": None,
            }
        )
        next_state = state.model_copy(
            update={"step": record.step, "actions": [*state.actions, action]}
        )

        def write() -> None:
            _atomic(
                self.folder / tree_path,
                _json(trees[0] if len(trees) == 1 else {"pre_states": trees}),
            )
            self._checkpoint(next_state)

        await asyncio.to_thread(write)
        self._state, self._pre_states = next_state, trees

    def _checkpoint(self, state: AgentState) -> None:
        # A later step can reveal that earlier free text was sensitive. Publish only
        # structural metadata; do not try to scrub previous checkpoints after the fact.
        payload = state.model_dump(
            include={"step", "max_steps", "repair_attempts", "output_attempts"}
        )
        payload["actions"] = [
            {
                **a.model_dump(include={"step", "name", "ok", "screen_changed"}),
                "ui_state": f"ui_states/{a.step:04}.json",
                "screenshot": (
                    str(a.before.screenshot.path)
                    if a.before.screenshot is not None
                    and a.before.screenshot.path is not None
                    and not run_sensitive_values.get()
                    else None
                ),
                "args": {},
                "summary": "[REDACTED]",
            }
            for a in state.actions
        ]
        _atomic(self.folder / f"checkpoints/{state.step:04}.json", _json(payload))

    async def finish(self, state: AgentState) -> None:
        def write() -> None:
            _atomic(self.macro_path, _json(macro_data(state.goal, state.actions)))

        await asyncio.to_thread(write)

"""Publish a compact state checkpoint after its observation files are on disk."""

import asyncio
import json
from itertools import chain, repeat
from pathlib import Path
from tempfile import NamedTemporaryFile

from mobilerun.macro.models import build_screen_snapshot
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.memory.snapshot import StepSnapshot
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
        self._ui_state: list[dict[str, JSONValue]] = []
        self._pre_states: list[dict[str, JSONValue]] = []
        self._guards: list[dict[str, JSONValue] | None] = []
        self._screenshot: bytes | None = None
        self._decision_count = 0

    async def on_decision(self, choices: list[dict[str, JSONValue]]) -> str:
        """Write one redacted option list per Jev call, including retries."""
        self._decision_count += 1
        path = f"decisions/{self._decision_count:04}.json"
        await asyncio.to_thread(_atomic, self.folder / path, _json({"choices": choices}))
        return path

    async def on_observe(
        self, snapshot: StepSnapshot, screenshot: bytes | None, ui_state: list[dict[str, JSONValue]]
    ) -> None:
        """Pin this observation for subsequent actions at the same step."""
        state = self._state
        state.goal = snapshot.goal
        if state.step != snapshot.step:
            self._pre_states.clear()
        state.step = snapshot.step
        self._state = state
        self._ui_state, self._screenshot = ui_state, screenshot

    async def on_step(
        self,
        record: ActionRecord,
        pre_state: dict[str, JSONValue] | None = None,
        *,
        guard_redacted: bool = False,
    ) -> None:
        validate_vault_reference(record)
        state = self._state
        if state.step == 0 or state.step != record.step:
            raise ValueError("Action requires an observation from the same step")
        image = self._screenshot
        tree_path = f"ui_states/{record.step:04}.json"
        image_path = f"screenshots/{record.step:04}.png" if image is not None else None
        guard = pre_state if pre_state is not None else build_screen_snapshot(self._ui_state)
        trees = [*self._pre_states, guard]
        action = record.model_copy(update={"ui_state": tree_path, "screenshot": image_path})
        next_state = state.model_copy(update={"actions": [*state.actions, action]})

        def write() -> None:
            _atomic(
                self.folder / tree_path,
                _json(trees[0] if len(trees) == 1 else {"pre_states": trees}),
            )
            if image_path is not None and image is not None:
                _atomic(self.folder / image_path, image)
            self._checkpoint(next_state)

        await asyncio.to_thread(write)
        self._state, self._pre_states = next_state, trees
        self._guards.append(None if guard_redacted else guard)

    def _checkpoint(self, state: AgentState) -> None:
        # A later step can reveal that earlier free text was sensitive. Publish only
        # structural metadata; do not try to scrub previous checkpoints after the fact.
        payload = state.model_dump(
            include={"step", "max_steps", "repair_attempts", "output_attempts"}
        )
        payload["actions"] = [
            {
                **a.model_dump(include={"step", "name", "ok", "screenshot", "ui_state"}),
                "args": ({"text": "[REDACTED]"} if a.name == "input_text" else {}),
                "summary": "[REDACTED]",
            }
            for a in state.actions
        ]
        _atomic(self.folder / f"checkpoints/{state.step:04}.json", _json(payload))

    async def finish(self, state: AgentState) -> None:
        recorded = [(a.step, a.name) for a in self._state.actions]
        if recorded != [(a.step, a.name) for a in state.actions[: len(recorded)]]:
            raise ValueError("Final action history differs from the recorded steps")

        # Final actions that were not streamed have no observation guard.
        actions = zip(state.actions, chain(self._guards, repeat(None)), strict=False)

        def write() -> None:
            _atomic(self.macro_path, _json(macro_data(state.goal, actions)))

        await asyncio.to_thread(write)

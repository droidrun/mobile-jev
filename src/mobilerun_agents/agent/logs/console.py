"""Human-readable, log-safe output for the agent workflow event stream."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from time import monotonic
from typing import Any

from workflows.events import Event

from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.state import ActionRecord
from mobilerun_agents.errors import SelectorCannotServe, UnknownModel

from .diagnostics import ActionDetail, Detail, ModelCall, Screen, private_values
from .diagnostics import safe_text as _safe

_LOGGER = logging.getLogger(__name__)
_MEMORY = re.compile(r"<add_memory\b[^>]*>.*?(?:</add_memory\s*>|$)", re.I | re.S)


def _display_path(path: str | Path) -> str:
    location = Path(path)
    home = Path.home()
    if location.is_relative_to(home):
        location = Path("~") / location.relative_to(home)
    return _safe(str(location), limit=None)


class ConsoleLog:
    """Format workflow events without exposing action arguments or screenshots."""

    def __init__(self) -> None:
        self._step: int | None = None
        self._attempt = 0
        self._actions: list[str] = []
        self._details: list[str] = []
        self._started = monotonic()
        self._tokens: int | None = 0
        self._executor: str | None = None

    def on_event(self, event: Event) -> None:
        if private_values.get() is None:
            return
        match event:
            case Screen():
                self._attempt += 1
                _LOGGER.debug("%s · attempt %s", (self._step or 0) + 1, self._attempt)
                app = _safe(event.package or "?")
                location = f"{app}/{_safe(event.activity)}" if event.activity else app
                _LOGGER.debug(
                    "  screen: %s · %s elements",
                    location,
                    event.elements,
                )
            case ModelCall():
                image = f"{event.image[0]}×{event.image[1]}" if event.image else "no image"
                _LOGGER.debug("  image: %s", image)
                counts = " ".join(
                    f"{k}={v if v is not None else '?'}"
                    for k, v in (
                        ("input", event.inputs),
                        ("output", event.outputs),
                        ("reasoning", event.reasoning),
                    )
                )
                _LOGGER.debug("  LLM: %s · %.3fs · %s", _safe(event.name), event.seconds, counts)
                self._tokens = (
                    self._tokens + event.total
                    if self._tokens is not None and event.total is not None
                    else None
                )
            case Detail():
                if event.section == "planning":
                    if event.text.startswith("executor: "):
                        if self._executor == event.text:
                            return
                        self._executor = event.text
                    else:
                        self._executor = None
                if event.section == "thought":
                    text = event.text.strip()
                    words = text.split()
                    midpoint = len(words) // 2
                    if len(words) % 2 == 0 and words[:midpoint] == words[midpoint:]:
                        event = event.model_copy(update={"text": " ".join(words[:midpoint])})
                label = "repair " if event.section == "repair" else f"{event.section}: "
                text = event.text
                if event.section == "thought":
                    text = _safe(_MEMORY.sub(" ", text), 300)
                    # The producer sends at most 301 characters, before memory markup is removed.
                    if len(event.text) == 301 and not text.endswith("…"):
                        text += "…"
                else:
                    text = _safe(text, 300)
                if text:
                    _LOGGER.debug("  %s%s", label, text)
            case ActionDetail():
                if event.success and event.summary.endswith(" → None"):
                    self._details.append(f"result: success · {event.seconds:.3f}s")
                    return
                args = json.dumps(event.args, ensure_ascii=False)
                self._details.append(
                    f"selection: {event.wire} → {event.registry} {_safe(args, 400)}"
                )
                self._details.append(
                    f"result: {'success' if event.success else 'failure'} · {_safe(event.summary)} · {event.seconds:.3f}s"
                )

    def on_repair(self) -> None:
        if private_values.get() is not None:
            return
        self.flush_step()
        # Parse errors can contain the raw model response and tool parameters.
        _LOGGER.info("Retrying invalid selection", extra={"color": "dim"})

    def on_action(self, record: ActionRecord) -> None:
        if self._step != record.step:
            self.flush_step()
        self._step = record.step
        self._attempt = 0
        detail = ""
        if record.summary == "[REDACTED]":
            detail = " ***"
        elif record.name == "start_app":
            identifier = record.args.get("package")
            match identifier:
                case str():
                    detail = " " + _safe(identifier)
        elif record.name == "tap_element":
            match record.args.get("index"):
                case int() as index if type(index) is int:
                    detail = f" [{index}]"
        self._actions.append(f"{_safe(record.name)}{detail} {'✓' if record.ok else '✗'}")
        if private_values.get() is not None:
            self.flush_step()

    def on_subgoal(self, text: str) -> None:
        self.flush_step()
        _LOGGER.info("→ subgoal: %s", _safe(text))

    def flush_step(self) -> None:
        if self._actions:
            _LOGGER.info("%s · %s", self._step, "; ".join(self._actions))
            self._actions.clear()
            self._flush_details()

    def _flush_details(self) -> None:
        for detail in self._details:
            _LOGGER.debug("  %s", _safe(detail, None))
        self._details.clear()

    def on_result(
        self, result: AgentResult[Any], folder: str | None = None, macro_path: Path | None = None
    ) -> None:
        message = result.reason
        if result.answer and (result.success or result.reason == "complete"):
            message = result.answer
        self._finish(result.success, message, result.steps, folder, macro_path=macro_path)

    def on_error(self, error: BaseException, steps: int, folder: str | None = None) -> None:
        # Provider exceptions can include full requests; never log their raw text.
        message = (
            str(error)
            if type(error) in (SelectorCannotServe, UnknownModel)
            else type(error).__name__
        )
        self._finish(False, message, steps, folder)

    def _finish(
        self,
        success: bool,
        message: str,
        steps: int,
        folder: str | None,
        macro_path: Path | None = None,
    ) -> None:
        self.flush_step()
        display_folder = _display_path(folder) if folder else None
        debug = private_values.get() is not None
        if debug:
            self._flush_details()
            _LOGGER.debug(
                "  end: total tokens=%s · total time=%.3fs%s",
                self._tokens if self._tokens is not None else "?",
                monotonic() - self._started,
                f" · run folder={display_folder}" if display_folder else "",
            )
            if macro_path is not None:
                _LOGGER.debug("  macro: %s", _display_path(macro_path))
        _LOGGER.info(
            "%s %s (%s %s%s)",
            "✓" if success else "✗",
            _safe(message, None if debug else 200),
            steps,
            "step" if steps == 1 else "steps",
            f" · {display_folder}" if display_folder else "",
            extra={"agent_result": True},
        )

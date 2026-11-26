"""Human-readable, log-safe output for the agent workflow event stream."""

from __future__ import annotations

import asyncio
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

from .diagnostics import ActionDetail, Detail, ModelCall, Retry, Screen, private_values
from .diagnostics import safe_text as _safe

_LOGGER = logging.getLogger(__name__)
_MEMORY = re.compile(r"<add_memory\b[^>]*>.*?(?:</add_memory\s*>|$)", re.I | re.S)


def _distinct_thought(text: str) -> str:
    normalized = " ".join(text.split())
    words = normalized.split()
    if len(words) % 2 == 0 and words[: len(words) // 2] == words[len(words) // 2 :]:
        return " ".join(words[: len(words) // 2])
    # Remove the longest repeated phrase without chopping off surrounding text.
    for length in range(len(words) // 2, 2, -1):
        for first in range(len(words) - length):
            phrase = words[first : first + length]
            for second in range(first + length, len(words) - length + 1):
                if words[second : second + length] == phrase:
                    return " ".join(words[:second] + words[second + length :])
    return normalized


def _display_path(path: str | Path) -> str:
    location = Path(path)
    home = Path.home()
    if location.is_relative_to(home):
        location = Path("~") / location.relative_to(home)
    return _safe(str(location), limit=None)


class ToolDetail(Detail):
    """A tool expansion message with a stable key across child workflows."""

    key: str
    count: int


class ConsoleLog:
    """Format workflow events without exposing action arguments or screenshots."""

    def __init__(self) -> None:
        self._step: int | None = None
        self._actions: list[tuple[str, list[str]]] = []
        self._details: list[str] = []
        self._delegated = False
        self._started = monotonic()
        self._tokens: int | None = 0
        self._executor: str | None = None
        self._tools: dict[str, int] = {}

    def on_event(self, event: Event) -> None:
        if private_values.get() is None:
            return
        match event:
            case Retry(attempt=attempt, reason=reason):
                _LOGGER.debug("  retry %s · %s", attempt, _safe(reason, 300))
            case Screen():
                self.flush_step()
                app = _safe(event.package) if event.package else "unknown app"
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
                if type(event) is ToolDetail:
                    if self._tools.get(event.key) == event.count:
                        return
                    self._tools[event.key] = event.count
                if event.section == "planning":
                    if event.text.startswith("executor: "):
                        if self._executor == event.text:
                            return
                        self._executor = event.text
                    else:
                        self._executor = None
                text = event.text
                if event.section == "thought":
                    text = _safe(_distinct_thought(_MEMORY.sub(" ", text)), 300)
                elif event.section == "tools":
                    text = _safe(text, limit=None)
                else:
                    text = _safe(text, 300)
                if text:
                    _LOGGER.debug("  %s: %s", event.section, text)
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

    def on_action(self, record: ActionRecord, delegated: bool = False) -> None:
        if self._step != record.step:
            self.flush_step()
        self._step = record.step
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
            label = record.args.get("label")
            if isinstance(label, str):
                detail += f" {_safe(label, 40)}"
        elif record.name == "tap":
            x, y = record.args.get("x"), record.args.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                detail = f" ({x:g}, {y:g})"
        elif record.name == "input_text":
            text = record.args.get("text")
            if isinstance(text, str):
                detail = f' "{_safe(text, 40)}"'
        self._delegated |= delegated
        self._actions.append(
            (f"{_safe(record.name)}{detail} {'✓' if record.ok else '✗'}", self._details)
        )
        self._details = []

    def on_subgoal(self, text: str) -> None:
        self.flush_step()
        _LOGGER.info("→ subgoal: %s", _safe(text))

    def flush_step(self) -> None:
        marker = " · llm" if self._delegated else ""
        for action, details in self._actions:
            _LOGGER.info("%s · %s%s", self._step, action, marker)
            for detail in details:
                _LOGGER.debug("  %s", _safe(detail, None))
        self._actions.clear()
        self._delegated = False

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
            "cancelled"
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else (
                str(error)
                if type(error) in (SelectorCannotServe, UnknownModel)
                else type(error).__name__
            )
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

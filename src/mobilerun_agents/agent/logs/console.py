"""Render workflow events at the console boundary."""

import logging
from collections.abc import Callable
from pathlib import Path

from workflows.events import Event

from mobilerun_agents.agent.events import (
    Acted,
    Detail,
    Finished,
    Observed,
    Repaired,
    Selected,
    Subgoal,
)

from .memory import MemoryDetail

_log = logging.getLogger(__name__)


class ConsoleLog:
    def __init__(self, redact: Callable[[str], str]) -> None:
        self._redact = redact
        self._step = 1
        self._memory_selectors: dict[int, set[str]] = {}

    def _line(self, text: str) -> str:
        return " ".join(self._redact(text).split())

    def event(self, event: Event) -> None:
        match event:
            case Observed(step=step, package=package, activity=activity, planner=planner):
                self._step = step
                _log.debug(
                    "  %sscreen: %s",
                    "planner " if planner else "",
                    self._line("/".join(p for p in (package, activity) if p)),
                )
            case Selected(thought=thought) if thought:
                _log.debug("  thought: %s", self._line(thought))
            case Acted(action=action):
                call = action.call
                _log.info(
                    "%s · %s %s", action.step, self._line(call.tool_name), "✓" if action.ok else "✗"
                )
                if _log.isEnabledFor(logging.DEBUG):
                    _log.debug(
                        "  arguments: %s · result: %s",
                        self._line(str(call.tool_kwargs)),
                        self._line(action.summary),
                    )
            case Repaired(attempt=attempt, reason=reason):
                _log.info("Retrying invalid selection")
                _log.debug("  retry %s · %s", attempt, self._line(reason))
            case Subgoal(text=text, step=step, budget=budget):
                self._memory_selectors.pop(self._step, None)
                progress = (
                    f" (step {step}/{budget})" if step is not None and budget is not None else ""
                )
                _log.info("→ subgoal%s: %s", progress, self._line(text))
            case MemoryDetail() if _log.isEnabledFor(logging.DEBUG):
                self._memory(event)
            case MemoryDetail():
                pass
            case Detail(section=section, text=text):
                _log.debug("  %s: %s", self._line(section), self._line(text))

    def _memory(self, detail: MemoryDetail) -> None:
        selectors = self._memory_selectors.setdefault(self._step, set())
        later = bool(selectors)
        selectors.add(detail.selector)
        if later:
            _log.debug(
                "  memory (%s): %s",
                self._line(detail.selector),
                self._line(detail.counts),
                extra={"agent_memory": True},
            )
            return
        counts = []
        previews = []
        for name, text in detail.blocks.items():
            lines = self._redact(text).splitlines()
            n = len(lines)
            safe_name = self._line(name)
            counts.append(f"{safe_name}({n}{' lines' if n > 1 else ''})")
            first = lines[0].strip() if lines else ""
            if first:
                preview = first if len(first) <= 120 else first[:119] + "…"
                previews.append(f"    {safe_name}: {preview}")
        if detail.last_steps:
            previews.append(f"    last steps: {self._line(' · '.join(detail.last_steps))}")
        header = f"  memory: {self._line(detail.counts)} · blocks: {', '.join(counts) or '(none)'}"
        _log.debug("%s", "\n".join([header, *previews]), extra={"agent_memory": True})

    def subgoal_finished(self, success: bool, message: str) -> None:
        _log.info("%s subgoal: %s", "✓" if success else "✗", self._line(message))

    def finish(
        self, success: bool, message: str, steps: int | Finished, folder: Path | None = None
    ) -> None:
        breakdown = ""
        match steps:
            case Finished(steps=count, budget=budget, planner_turns=turn) if (
                budget is not None and turn is not None
            ):
                step_count = f"{count}/{budget}"
                unit = "steps"
                breakdown = f" · {turn} planner, {count - turn} actions"
            case Finished(steps=count) | (int() as count):
                step_count = str(count)
                unit = "step" if count == 1 else "steps"
        _log.info(
            "%s %s (%s %s%s%s)",
            "✓" if success else "✗",
            self._line(message),
            step_count,
            unit,
            breakdown,
            f" · {self._line(str(folder))}" if folder else "",
            extra={"agent_result": True},
        )

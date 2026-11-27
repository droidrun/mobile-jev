"""Render workflow events at the console boundary."""

import logging
import os
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


def _line(text: str) -> str:
    return " ".join(text.split())


class ConsoleLog:
    def __init__(self, folder: Path | None = None) -> None:
        self._step = 1
        self._memory_selectors: dict[int, set[str]] = {}
        self._memory_folder = (
            folder / "memory" if folder and _log.isEnabledFor(logging.DEBUG) else None
        )
        if self._memory_folder is not None:
            self._memory_folder.mkdir(parents=True, mode=0o700, exist_ok=True)
            _log.debug("Memory details: %s", _line(str(self._memory_folder)))

    def event(self, event: Event) -> None:
        match event:
            case Observed(step=step, package=package, activity=activity, planner=planner):
                self._step = step
                _log.debug(
                    "  %sscreen: %s",
                    "planner " if planner else "",
                    _line("/".join(p for p in (package, activity) if p)),
                )
            case Selected(thought=thought) if thought:
                _log.debug("  thought: %s", _line(thought))
            case Acted(action=action):
                call = action.call
                _log.info(
                    "%s · %s %s", action.step, _line(call.tool_name), "✓" if action.ok else "✗"
                )
                _log.debug("  arguments: %s · result: %s", call.tool_kwargs, _line(action.summary))
            case Repaired(attempt=attempt, reason=reason):
                _log.info("Retrying invalid selection")
                _log.debug("  retry %s · %s", attempt, _line(reason))
            case Subgoal(text=text, step=step, budget=budget):
                self._memory_selectors.pop(self._step, None)
                progress = (
                    f" (step {step}/{budget})" if step is not None and budget is not None else ""
                )
                _log.info("→ subgoal%s: %s", progress, _line(text))
            case MemoryDetail() if _log.isEnabledFor(logging.DEBUG):
                self._memory(event)
            case MemoryDetail():
                pass
            case Detail(section=section, text=text):
                _log.debug("  %s: %s", _line(section), _line(text))

    def _memory(self, detail: MemoryDetail) -> None:
        selectors = self._memory_selectors.setdefault(self._step, set())
        later = bool(selectors)
        selectors.add(detail.selector)
        if later:
            _log.debug(
                "  memory (%s): %s",
                _line(detail.selector),
                _line(detail.counts),
                extra={"agent_memory": True},
            )
            self._write_memory(detail, detail.selector)
            return
        counts = []
        previews = []
        for name, text in detail.blocks.items():
            n = len(text.splitlines())
            counts.append(f"{name}({n}{' lines' if n > 1 else ''})")
            first = next(iter(text.splitlines()), "").strip()
            if first:
                preview = first if len(first) <= 120 else first[:119] + "…"
                previews.append(f"    {_line(name)}: {preview}")
        if detail.last_steps:
            previews.append(f"    last steps: {_line(' · '.join(detail.last_steps))}")
        header = f"  memory: {detail.counts} · blocks: {_line(', '.join(counts) or '(none)')}"
        _log.debug("%s", "\n".join([header, *previews]), extra={"agent_memory": True})
        self._write_memory(detail)

    def _write_memory(self, detail: MemoryDetail, selector: str | None = None) -> None:
        if self._memory_folder is not None:
            suffix = f"-{_line(selector)}" if selector else ""
            path = self._memory_folder / f"{self._step:04d}{suffix}.txt"
            if path.exists():
                index = 2
                while path.exists():
                    path = self._memory_folder / f"{self._step:04d}{suffix}-{index}.txt"
                    index += 1
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            descriptor = os.open(path, flags, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
            except BaseException:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(detail.text + "\n")

    def subgoal_finished(self, success: bool, message: str) -> None:
        _log.info("%s subgoal: %s", "✓" if success else "✗", _line(message))

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
            _line(message),
            step_count,
            unit,
            breakdown,
            f" · {_line(str(folder))}" if folder else "",
            extra={"agent_result": True},
        )

"""Render workflow events at the console boundary."""

import logging
import os
from pathlib import Path

from workflows.events import Event

from mobilerun_agents.agent.events import (
    Acted,
    Detail,
    Observed,
    Repaired,
    Selected,
    Subgoal,
)

from .memory import MemoryDetail
from .redaction import Redactor

_log = logging.getLogger(__name__)


class ConsoleLog:
    def __init__(self, redact: Redactor, folder: Path | None = None) -> None:
        self._redact = redact
        self._step = 1
        self._memory_selectors: dict[int, set[str]] = {}
        self._memory_folder = (
            folder / "memory" if folder and _log.isEnabledFor(logging.DEBUG) else None
        )
        self._memory_revision = redact.revision
        if self._memory_folder is not None:
            self._memory_folder.mkdir(parents=True, mode=0o700, exist_ok=True)
            _log.debug("Memory details: %s", self._line(str(self._memory_folder)))

    def _line(self, text: str) -> str:
        return " ".join(self._redact.text(text).split())

    def event(self, event: Event) -> None:
        match event:
            case Observed(step=step, package=package, activity=activity):
                self._step = step
                _log.debug(
                    "  screen: %s", self._line("/".join(p for p in (package, activity) if p))
                )
            case Selected(thought=thought) if thought:
                _log.debug("  thought: %s", self._line(thought))
            case Acted(action=action):
                call = action.call
                args = self._redact.value(call.tool_kwargs)
                _log.info(
                    "%s · %s %s", action.step, self._line(call.tool_name), "✓" if action.ok else "✗"
                )
                _log.debug("  arguments: %s · result: %s", args, self._line(action.summary))
            case Repaired(attempt=attempt, reason=reason):
                _log.info("Retrying invalid selection")
                _log.debug("  retry %s · %s", attempt, self._line(reason))
            case Subgoal(text=text):
                self._memory_selectors.pop(self._step, None)
                _log.info("→ subgoal: %s", self._line(text))
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
            self._write_memory(detail, detail.selector)
            return
        counts = []
        previews = []
        for name, text in detail.blocks.items():
            n = len(text.splitlines())
            counts.append(f"{name}({n}{' lines' if n > 1 else ''})")
            # Redact before clipping, so a long secret cannot leak a prefix.
            first = next(iter(self._redact.text(text).splitlines()), "").strip()
            if first:
                preview = first if len(first) <= 120 else first[:119] + "…"
                previews.append(f"    {self._line(name)}: {preview}")
        if detail.last_steps:
            previews.append(f"    last steps: {self._line(' · '.join(detail.last_steps))}")
        header = f"  memory: {detail.counts} · blocks: {self._line(', '.join(counts) or '(none)')}"
        _log.debug("%s", "\n".join([header, *previews]), extra={"agent_memory": True})
        self._write_memory(detail)

    def _write_memory(self, detail: MemoryDetail, selector: str | None = None) -> None:
        if self._memory_folder is not None:
            self._refresh_memory_redaction()
            suffix = f"-{self._line(selector)}" if selector else ""
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
                file.write(self._redact.text(detail.text) + "\n")

    def _refresh_memory_redaction(self) -> None:
        if self._memory_folder is None or self._memory_revision == self._redact.revision:
            return
        for path in self._memory_folder.glob("*.txt"):
            path.write_text(self._redact.text(path.read_text(encoding="utf-8")), encoding="utf-8")
        self._memory_revision = self._redact.revision

    def subgoal_finished(self, success: bool, message: str) -> None:
        _log.info("%s subgoal: %s", "✓" if success else "✗", self._line(message))

    def finish(self, success: bool, message: str, steps: int, folder: Path | None = None) -> None:
        self._refresh_memory_redaction()
        _log.info(
            "%s %s (%s %s%s)",
            "✓" if success else "✗",
            self._line(message),
            steps,
            "step" if steps == 1 else "steps",
            f" · {self._line(str(folder))}" if folder else "",
            extra={"agent_result": True},
        )

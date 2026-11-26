"""Run-local diagnostics. Selectors publish metadata, never replies or image bytes."""

import re
from collections.abc import Callable
from contextvars import ContextVar
from typing import Literal

from mobilerun.models.context import JSONValue
from workflows.events import Event


class DebugStep(Event):
    """Base type routed exclusively to the debug console."""


class Screen(DebugStep):
    package: str | None
    activity: str | None
    elements: int


class Retry(DebugStep):
    attempt: int
    reason: str


class ModelCall(DebugStep):
    name: str
    image: tuple[int, int] | None
    seconds: float
    inputs: int | None
    outputs: int | None
    reasoning: int | None
    total: int | None


class Detail(DebugStep):
    section: Literal[
        "selector",
        "thought",
        "memory",
        "planning",
        "image",
        "jev input",
        "jev choices",
        "jev answer",
        "jev abstain",
        "tools",
    ]
    text: str


class ActionDetail(DebugStep):
    wire: str
    registry: str
    args: dict[str, JSONValue]
    success: bool
    summary: str
    seconds: float


class DecisionOptions(Event):
    """Redacted Jev options for a checkpoint sidecar, not console output."""

    summary: str
    choices: list[dict[str, JSONValue]]


sink: ContextVar[Callable[[DebugStep], None] | None] = ContextVar("model_diagnostics", default=None)
decision_sink: ContextVar[Callable[[DecisionOptions], None] | None] = ContextVar(
    "decision_options", default=None
)
record_decisions: ContextVar[bool] = ContextVar("record_decisions", default=False)
private_values: ContextVar[set[str] | None] = ContextVar("debug_private_values", default=None)


def protect(value: JSONValue, hidden: set[str] | None = None) -> None:
    if hidden is None:
        hidden = private_values.get()
    if hidden is not None:
        match value:
            case list():
                for item in value:
                    protect(item, hidden)
            case dict():
                for item in value.values():
                    protect(item, hidden)
            case None:
                pass
            case _:
                if str(value):
                    hidden.add(str(value))


_KEY = re.compile(
    r"\b(?:sk-[\w-]{8,}|AIza[\w-]{20,}|Bearer\s+[^\s,;]+|data:image/[^\s]+|iVBORw0KGgo[A-Za-z0-9+/=]+)|/9j/[A-Za-z0-9+/=]+",
    re.I,
)
_SENSITIVE = re.compile(
    r"(?i)\b(api[_ -]?key|(?:access[_ -]?)?token|password|secret)([\"']?)\s*[:=]\s*"
    r"""(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;]+)"""
)


def redact(text: str, hidden: set[str] | None = None) -> str:
    if hidden is None:
        hidden = private_values.get()
    for secret in sorted(hidden or (), key=len, reverse=True):
        text = text.replace(secret, "***")
    return _KEY.sub("***", text)


def safe_text(text: str, limit: int | None = 200) -> str:
    text = redact(text).replace("[REDACTED]", "***")
    text = " ".join("".join(char if char.isprintable() else " " for char in text).split())
    cleaned = _SENSITIVE.sub(lambda m: f'{m[1]}{m[2]}: "***"' if m[2] else f"{m[1]}=***", text)
    if limit is not None and len(cleaned) > limit:
        cleaned = f"{cleaned[:limit]}…"
    return cleaned

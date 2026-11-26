"""Mask tracked sensitive values at run-owned output boundaries."""

import json
from contextvars import ContextVar

from mobilerun.models.context import JSONValue

from .diagnostics import private_values, protect, redact

run_sensitive_values: ContextVar[set[str] | None] = ContextVar(
    "agent_sensitive_values", default=None
)


def protect_run_value(value: JSONValue) -> None:
    hidden = run_sensitive_values.get()
    assert hidden is not None
    debug_values = private_values.get()
    protect(value, hidden)
    if debug_values is not None:
        debug_values.update(hidden)


def redact_run_text(text: str) -> str:
    hidden = run_sensitive_values.get()
    if not hidden:
        return text
    variants = {
        encoded
        for value in hidden
        for encoded in (
            value,
            json.dumps(value)[1:-1],
            json.dumps(value, ensure_ascii=False)[1:-1],
            repr(value)[1:-1],
            " ".join(value.split()),  # Base replay guards normalize UI whitespace.
        )
        if encoded
    }
    return redact(text, variants)


def redact_run_value(value: JSONValue) -> JSONValue:
    match value:
        case str():
            return redact_run_text(value)
        case list():
            return [redact_run_value(item) for item in value]
        case dict():
            return redact_run_arguments(value)
        case _:
            return "***" if str(value) in (run_sensitive_values.get() or ()) else value


def redact_run_arguments(args: dict[str, JSONValue]) -> dict[str, JSONValue]:
    return {redact_run_text(key): redact_run_value(value) for key, value in args.items()}

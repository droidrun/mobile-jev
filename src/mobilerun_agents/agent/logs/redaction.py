"""Mask tracked sensitive values at run-owned output boundaries."""

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from urllib.parse import quote, quote_plus
from weakref import WeakKeyDictionary

from mobilerun.core.credentials.base import CredentialProvider
from mobilerun.core.devices import Device
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


# Base's secret scope is task-local and expires before a vault action returns.
# Observe only the explicit input boundary, never the ordinary text path.
_input_observers: WeakKeyDictionary[Device, tuple[int, object]] = WeakKeyDictionary()


@contextmanager
def _observe_secret_input(device: Device) -> Iterator[None]:
    active = _input_observers.get(device)
    if active is None:
        previous = device.__dict__.get("_input_secret")
        original = device._input_secret

        async def tracked(value: str) -> bool:
            if run_sensitive_values.get() is not None:
                protect_run_value(value)
            return await original(value)

        device.__dict__["_input_secret"] = tracked
        _input_observers[device] = (1, previous)
    else:
        count, previous = active
        _input_observers[device] = (count + 1, previous)
    try:
        yield
    finally:
        count, previous = _input_observers[device]
        if count > 1:
            _input_observers[device] = (count - 1, previous)
        else:
            del _input_observers[device]
            if previous is None:
                device.__dict__.pop("_input_secret")
            else:
                device.__dict__["_input_secret"] = previous


@contextmanager
def observe_secret_inputs(device: Device) -> Iterator[None]:
    if run_sensitive_values.get() is None:
        yield
        return
    targets = {device, device._unwrap()}
    for extension in device.extensions.values():
        match extension:
            case CredentialProvider(device=bound) if bound is not None:
                targets.add(bound)
    with ExitStack() as scopes:
        for target in targets:
            scopes.enter_context(_observe_secret_input(target))
        yield


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
            quote(value, safe="", errors="surrogatepass"),
            quote_plus(value, safe="", errors="surrogatepass"),
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

"""Run-local redaction for console, checkpoints and export."""

import json
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any
from urllib.parse import quote, quote_plus

from mobilerun.core.credentials.base import CredentialProvider
from mobilerun.core.devices import Device

MASK = "***"
_KEY = re.compile(r"\b(?:sk-[\w-]{8,}|AIza[\w-]{20,}|Bearer\s+[^\s,;]+|data:image/[^\s]+)", re.I)


class Redactor:
    def __init__(self) -> None:
        self._values: set[str] = set()
        self.revision = 0

    def add(self, value: object) -> None:
        match value:
            case dict():
                for key, item in value.items():
                    self.add(key)
                    self.add(item)
            case list() | tuple():
                for item in value:
                    self.add(item)
            case str() | int() | float() | bool():
                text = str(value)
                if text and text not in self._values:
                    self._values.update(
                        filter(
                            None,
                            (
                                text,
                                " ".join(text.split()),
                                json.dumps(text)[1:-1],
                                repr(text)[1:-1],
                                quote(text, safe="", errors="surrogatepass"),
                                quote_plus(text, safe="", errors="surrogatepass"),
                            ),
                        )
                    )
                    self.revision += 1

    def _registered_pattern(self) -> re.Pattern[str] | None:
        if not self._values:
            return None
        return re.compile("|".join(map(re.escape, sorted(self._values, key=len, reverse=True))))

    def text(self, value: str) -> str:
        pattern = self._registered_pattern()
        if pattern is not None:
            value = pattern.sub(lambda _: MASK, value)
        return _KEY.sub(MASK, value)

    def masked_spans(self, value: str) -> list[tuple[int, int]]:
        """Find registered and key matches on the original text, retaining overlaps."""
        spans = [match.span() for match in _KEY.finditer(value)]
        pattern = self._registered_pattern()
        if pattern is not None:
            spans.extend(match.span() for match in pattern.finditer(value))
        return sorted(spans)

    def value(self, value: Any, freeform: bool = True) -> Any:
        match value:
            case str():
                return self.text(value)
            case list() | tuple():
                return [self.value(item, freeform) for item in value]
            case dict():
                return {
                    (self.text(key) if freeform else key): (
                        self.value(item) if freeform else self._checkpoint_field(key, item)
                    )
                    for key, item in value.items()
                    if key != "screenshot" or not self.revision or freeform
                }
            case _:
                return MASK if freeform and str(value) in self._values else value

    def _checkpoint_field(self, key: str, value: Any) -> Any:
        match key, value:
            case "at", _:
                return value
            case "device_context", list() as items:
                return [self._context_item(item) for item in items]
            case _:
                return self.value(
                    value, freeform=key in {"tool_kwargs", "elements", "phone_state", "output"}
                )

    def _context_item(self, item: object) -> object:
        match item:
            case dict():
                return {
                    name: self.value(value, freeform=name == "data") for name, value in item.items()
                }
            case _:
                return self.value(item, freeform=False)


@dataclass(eq=False)
class _VaultScope:
    targets: set[Device]
    redactor: Redactor
    active: bool = True


# The supported pinned base has no vault observation hook. Share the wrapper, not the redactor.
_vault_scopes: ContextVar[tuple[_VaultScope, ...]] = ContextVar("agent_vault_scopes", default=())
_vault_wrappers: dict[Device, tuple[int, ExitStack]] = {}
_vault_lock = Lock()


@contextmanager
def _track_vault_input(target: Device) -> Iterator[None]:
    with _vault_lock:
        if target not in _vault_wrappers:
            attributes = target.__dict__
            original = target._input_secret
            restore = ExitStack()
            if "_input_secret" in attributes:
                restore.callback(
                    attributes.__setitem__, "_input_secret", attributes["_input_secret"]
                )
            else:
                restore.callback(attributes.pop, "_input_secret", None)

            async def tracked(value: str) -> bool:
                for scope in reversed(_vault_scopes.get()):
                    if scope.active and target in scope.targets:
                        scope.redactor.add(value)
                        break
                return await original(value)

            attributes["_input_secret"] = tracked
            _vault_wrappers[target] = (0, restore)
        count, restore = _vault_wrappers[target]
        _vault_wrappers[target] = (count + 1, restore)
    try:
        yield
    finally:
        with _vault_lock:
            count, restore = _vault_wrappers[target]
            if count == 1:
                del _vault_wrappers[target]
                restore.close()
            else:
                _vault_wrappers[target] = (count - 1, restore)


@contextmanager
def observe_vault(device: Device, redact: Redactor) -> Iterator[None]:
    """Observe only the base vault's secret-input boundary, never regular typing."""
    targets = {device, device._unwrap()}
    for extension in device.extensions.values():
        match extension:
            case CredentialProvider(device=bound) if bound is not None:
                targets.add(bound)

    with ExitStack() as scope:
        for target in targets:
            scope.enter_context(_track_vault_input(target))
        registration = _VaultScope(targets, redact)
        _vault_scopes.set((*_vault_scopes.get(), registration))
        try:
            yield
        finally:
            registration.active = False
            _vault_scopes.set(
                tuple(item for item in _vault_scopes.get() if item is not registration)
            )

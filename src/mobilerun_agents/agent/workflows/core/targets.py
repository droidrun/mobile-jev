"""Resolve action targets and count repeated actions or screens."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby, takewhile
from math import isfinite
from typing import Any

from mobilerun.core.state.ui_state import UIState

from .state import ActionRecord

_COORDINATE_GRID = 48
_EDITABLE_TYPES = ("edittext", "textfield", "searchfield", "searchview", "xcuielementtypetextview")


def _resource(node: dict[str, Any]) -> str:
    return str(node.get("resourceId") or node.get("identifier") or "")


def _password(node: dict[str, Any]) -> bool:
    details = node.get("details") or {}
    return bool(
        node.get("password")
        or node.get("isPassword")
        or details.get("password")
        or details.get("isPassword")
        or any("securetextfield" in str(node.get(k, "")).lower() for k in ("type", "className"))
    )


def _editable(node: dict[str, Any]) -> bool:
    details = node.get("details") or {}
    kinds = (str(node.get(k, "")).lower() for k in ("type", "className"))
    # Android TextView is not editable; iOS TextView can be.
    return any(
        node.get(k) or details.get(k) for k in ("isEditable", "editable", "focused", "isFocused")
    ) or any(kind.endswith(_EDITABLE_TYPES) for kind in kinds)


def _focused_input(ui: UIState, node: dict[str, Any]) -> dict[str, Any] | None:
    match ui.phone_state.get("focusedElement"):
        case dict() as focused if focused:
            pass
        case _:
            return None
    if not (ui.phone_state.get("isEditable") or _editable(focused) or _password(focused)):
        return None
    if focused.get("index") is not None and focused["index"] == node.get("index"):
        return focused
    if (resource := _resource(focused)) and resource == _resource(node):
        return focused
    # Some providers only report focused text, with no identifier.
    if focused.get("text") and focused["text"] == node.get("text"):
        return focused
    return None


def element_label(
    node: dict[str, Any],
    limit: int | None = 60,
    mask: Callable[[str], str] = str,
    focus: dict[str, Any] | None = None,
) -> str:
    details = node.get("details") or {}
    class_name = str(node.get("className") or node.get("type") or "Element")
    short = class_name.rsplit(".", 1)[-1].removeprefix("XCUIElementType") or "Element"
    password = _password(node) or (focus is not None and _password(focus))
    editable = password or focus is not None or _editable(node)
    hint = (node.get("hint"), details.get("hint"), node.get("placeholder"))
    description: tuple[object, ...] = (
        () if password else (node.get("contentDescription"), details.get("description"))
    )
    if editable:
        description = tuple(
            item for item in description if item not in (node.get("text"), node.get("value"))
        )
    label = next(
        (
            " ".join(mask(value).split())
            for value in (
                *(hint if editable else (node.get("text"),)),
                *description,
                None if editable else node.get("label"),
                _resource(node),
            )
            if type(value) is str and value.strip() and value not in {class_name, short}
        ),
        "",
    )
    if limit is not None and len(label) > limit:
        label = label[: limit - 1] + "…"
    return f'{short} "{label}"' if label else short


@dataclass(frozen=True)
class ActionTarget:
    index: int | None = None
    point: tuple[int, int] | None = None
    node: dict[str, Any] | None = None
    focus: dict[str, Any] | None = None

    @property
    def fields(self) -> frozenset[str]:
        return frozenset({"index"} if self.index is not None else {"x", "y"})

    def label(self, limit: int | None = 60, mask: Callable[[str], str] = str) -> str:
        if self.node is not None:
            return element_label(self.node, limit, mask, self.focus)
        return str(self.point)

    def identity(self) -> tuple[object, ...]:
        if self.node is not None:
            return (
                self.node.get("className") or self.node.get("type"),
                _resource(self.node),
                self.label(limit=None),
                _bounds(self.node.get("bounds")),
            )
        assert self.point is not None
        return tuple(n // _COORDINATE_GRID for n in self.point)


def action_target(name: str, args: dict[str, Any], ui: UIState | None) -> ActionTarget | None:
    index = args.get("index")
    if type(index) is int and ui is not None and (node := ui.get_element(index)) is not None:
        return ActionTarget(index=index, node=node, focus=_focused_input(ui, node))
    if name.rsplit(".", 1)[-1] in {"tap", "long_press", "long_press_at"}:
        x, y = (_rounded(args.get(key)) for key in ("x", "y"))
        if x is not None and y is not None:
            point = (x, y)
            if ui is not None and (node := element_at(ui, point)) is not None:
                return ActionTarget(point=point, node=node, focus=_focused_input(ui, node))
            return ActionTarget(point=point)
    return None


def _rounded(value: object) -> int | None:
    if type(value) is int:
        return value
    return round(value) if type(value) is float and isfinite(value) else None


def target(record: ActionRecord, limit: int | None = 60) -> str | None:
    """Return the shared label, independent of the transient index and screen fingerprint."""
    resolved = _record_target(record)
    return resolved.label(limit) if resolved is not None else None


def _record_target(record: ActionRecord) -> ActionTarget | None:
    return action_target(record.call.tool_name, record.call.tool_kwargs, record.before.ui)


def repeated_target(actions: Sequence[ActionRecord]) -> int:
    """Count the consecutive calls matching the last action's name, target and other args."""
    if not actions:
        return 0

    def key(record: ActionRecord) -> tuple[str, tuple[object, ...], dict[str, Any]] | None:
        call = record.call
        resolved = _record_target(record)
        if resolved is None:
            return None
        args = {k: v for k, v in call.tool_kwargs.items() if k not in resolved.fields}
        return call.tool_name, resolved.identity(), args

    last = key(actions[-1])
    if last is None:
        return 0
    count = 1
    for i in range(len(actions) - 2, -1, -1):
        if key(actions[i]) != last:
            break
        count += 1
    return count


def unchanged_steps(actions: Sequence[ActionRecord]) -> int:
    """Count distinct steps in the trailing unchanged action records."""
    unchanged = takewhile(lambda record: record.screen_changed is False, reversed(actions))
    return len({record.step for record in unchanged})


def cycle_visits(actions: Sequence[ActionRecord], current: tuple[str, str] | None) -> int:
    """Count visits to the current screen since the last unknown or newly seen screen."""
    if current is None:
        return 0
    history = [record.before.fingerprint for record in actions] + [current]
    previous_screens: set[tuple[str, str]] = set()
    visits = 0
    for screen, _ in groupby(history):
        if screen is None:
            visits = 0
            continue
        if screen not in previous_screens:
            previous_screens.add(screen)
            visits = 0
        if screen == current:
            visits += 1
    return visits


def element_at(ui: UIState, point: tuple[int, int]) -> dict[str, Any] | None:
    """Resolve a unique visible, unblocked target without assuming list draw order."""
    x, y = point

    def contains(raw: object) -> bool:
        box = _bounds(raw)
        return box is not None and box[0] <= x < box[2] and box[1] <= y < box[3]

    hits = _hits(ui.elements, contains)
    first = next(hits, None)
    return first if next(hits, None) is None else None


def _hits(
    nodes: list[dict[str, Any]], contains: Callable[[object], bool]
) -> Iterator[dict[str, Any]]:
    for node in nodes:
        if node.get("isVisibleToUser") is False:
            continue
        if any(contains(blocker) for blocker in node.get("tapBlockers") or ()):
            continue
        children = _hits(node.get("children") or [], contains)
        if (child := next(children, None)) is not None:
            yield child
            yield from children
        elif contains(node.get("bounds")):
            yield node


def _bounds(raw: object) -> tuple[int, int, int, int] | None:
    if type(raw) is not str:
        return None
    try:
        left, top, right, bottom = (round(float(value)) for value in raw.split(","))
    except (ValueError, OverflowError):
        return None
    return left, top, right, bottom

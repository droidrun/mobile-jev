"""Build guarded base macros from successful actions in saved agent state."""

from collections.abc import Iterable
from typing import Any

from mobilerun.macro.models import MACRO_SCHEMA_VERSION, build_screen_snapshot

from mobilerun_agents.agent.logs.redaction import MASK, Redactor
from mobilerun_agents.agent.workflows.core.state import ActionRecord

_REPLAYABLE = {
    "back",
    "button_press",
    "open_url",
    "press_key_code",
    "start_app",
    "stop_app",
    "tap",
}


def _masked(value: object) -> bool:
    match value:
        case str():
            return MASK in value
        case dict():
            return any(_masked(key) or _masked(item) for key, item in value.items())
        case list():
            return any(map(_masked, value))
        case _:
            return False


def _primitive(record: ActionRecord) -> dict[str, Any] | None:
    name = record.call.tool_name
    args = record.call.tool_kwargs
    if name == "extension_call" and set(args) == {"name", "arguments"}:
        return {**args, "action_type": name}
    if name == "type_secret":
        if set(args) <= {"name", "field", "package"} and (args.get("name") or args.get("package")):
            return {**args, "action_type": name}
        return None
    if name == "press_button":
        return {"action_type": "button_press", "button": args.get("button")}
    if name == "wait":
        return {"action_type": "wait", "duration": args.get("duration", args.get("seconds", 1.0))}
    if name == "tap_element":
        match args.get("index"):
            case int() as index:
                try:
                    x, y = record.before.ui.get_element_coords(index)
                    return {"action_type": "tap", "x": x, "y": y}
                except (KeyError, ValueError):
                    return None
        return None
    if "." in name:
        return {"action_type": "extension_call", "name": name, "arguments": args}
    if name in {"swipe", "drag"}:
        match args:
            case {"x1": x1, "y1": y1, "x2": x2, "y2": y2}:
                timing = (
                    {"duration_ms": args.get("duration_ms", 1000)}
                    if name == "swipe"
                    else {"duration": args.get("duration", 3.0)}
                )
                return {
                    "action_type": name,
                    "start_x": x1,
                    "start_y": y1,
                    "end_x": x2,
                    "end_y": y2,
                    **timing,
                }
        return None
    if name == "input_text":
        # Keep the typing step; base replay reads only text and clear.
        return {"action_type": name, **{key: args[key] for key in ("text", "clear") if key in args}}
    if name in _REPLAYABLE:
        return {**args, "action_type": name}
    return None


def macro_data(goal: str, records: Iterable[ActionRecord], redact: Redactor) -> dict[str, Any]:
    actions = []
    for record in records:
        if not record.ok or (payload := _primitive(record)) is None:
            continue
        # A restored checkpoint already contains markers; never replay them as arguments.
        if redact.value(payload) != payload or _masked(payload):
            continue
        guard = build_screen_snapshot(record.before.ui)
        has_evidence = guard["nodes"] or any(guard["phone_state"].values())
        safe = has_evidence and redact.value(guard) == guard and not _masked(guard)
        actions.append({**payload, "pre_state": guard if safe else None})
    return {
        "description": redact.text(goal),
        "macro_schema_version": MACRO_SCHEMA_VERSION,
        "total_actions": len(actions),
        "actions": actions,
    }

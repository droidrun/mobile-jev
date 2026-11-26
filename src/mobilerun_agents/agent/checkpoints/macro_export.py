"""Export the *executed* base primitives, not model-facing tool selections."""

from collections.abc import Iterable

from mobilerun.macro.models import MACRO_SCHEMA_VERSION
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.logs.redaction import redact_run_arguments, redact_run_value
from mobilerun_agents.agent.workflows.core.state import ActionRecord


def validate_vault_reference(action: ActionRecord) -> None:
    if action.name == "type_secret" and action.args.keys() - {"name", "package", "field"}:
        raise ValueError("type_secret must contain only a vault reference")


def _redact_macro(
    fields: dict[str, JSONValue], actions: list[dict[str, JSONValue]]
) -> dict[str, object]:
    replayable = []
    for action in actions:
        masked = redact_run_arguments(action)
        guard = masked.pop("pre_state", None)
        original = {key: value for key, value in action.items() if key != "pre_state"}
        if masked != original:
            # Never replay altered arguments or action identifiers.
            continue
        # Base compares guard text with the live UI. A masked guard can never
        # reliably match, so omit it without dropping an otherwise safe step.
        replayable.append({**masked, "pre_state": guard if guard == action["pre_state"] else None})
    return {
        **{key: redact_run_value(value) for key, value in fields.items()},
        "macro_schema_version": MACRO_SCHEMA_VERSION,
        "total_actions": len(replayable),
        "actions": replayable,
    }


def macro_data(
    goal: str, actions_with_guards: Iterable[tuple[ActionRecord, dict[str, JSONValue] | None]]
) -> dict[str, object]:
    actions: list[dict[str, JSONValue]] = []
    for action, guard in actions_with_guards:
        validate_vault_reference(action)
        if not action.ok or not action.replayable:
            continue
        if (
            action.name in {"input_text", "set_clipboard"}
            and action.args.get("text") == "[REDACTED]"
        ):
            continue
        payload: dict[str, JSONValue] = (
            {"action_type": "extension_call", "name": action.name, "arguments": action.args}
            if "." in action.name
            else {**action.args, "action_type": action.name}
        )
        actions.append({**payload, "pre_state": guard})
    return _redact_macro({"description": goal}, actions)

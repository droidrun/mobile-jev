"""Expand a run's registry into fixed actions for choice-based selectors."""

from collections.abc import Sequence

from llama_index.core.tools import BaseTool

from mobilerun_agents.agent.logs.console import ToolDetail
from mobilerun_agents.agent.logs.diagnostics import safe_text, sink
from mobilerun_agents.agent.logs.redaction import redact_run_text
from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_prompt

from ..base import RegistryTool
from .exceptions import ATOMIC_EXCEPTIONS
from .schema import AtomicTool, from_schema, schema_summary

__all__ = ["expand_atomic_tools", "AtomicTool"]


def has_device_atomic_exception(member: BaseTool) -> bool:
    """Whether this native tool has a fixed-choice expansion instead of schema enumeration."""
    match member:
        case RegistryTool(action=action) if action and action.owner == "device":
            return (
                not action.sensitive_parameters and member.metadata.get_name() in ATOMIC_EXCEPTIONS
            )
    return False


def _changed(run: Run, name: str, count: int) -> bool:
    # Reuse the run's diagnostic cache. Track the latest count, not every count ever seen.
    prefix = f"{name}:"
    mark = f"{prefix}{count}"
    seen = run.atomic_drops
    if mark in seen:
        return False
    previous = next((key for key in seen if key.startswith(prefix)), None)
    if previous is not None:
        seen.remove(previous)
    seen.add(mark)
    return True


def _exception_summary(name: str, count: int) -> str:
    if name == "complete":
        return "done / report failure"
    noun = {"tap_element": "element", "start_app": "app", "swipe": "scroll direction"}[name]
    return f"{count} {noun}{'' if count == 1 else 's'}"


async def expand_atomic_tools(members: Sequence[BaseTool], run: Run) -> tuple[AtomicTool, ...]:
    options: list[AtomicTool] = []
    emit = sink.get()
    exceptions: list[tuple[str, int, str]] = []
    schemas: list[tuple[str, int, str]] = []
    passed: list[str] = []
    drops: list[tuple[str, int, str]] = []
    for member in sorted(
        members,
        key=lambda member: (member.metadata.get_name() == "complete", member.metadata.get_name()),
    ):
        name = member.metadata.get_name()
        if name == "delegate":
            options.append(
                AtomicTool(
                    name,
                    label=render_prompt(
                        run.config.decision.criteria_prompt, kind="delegate"
                    ),
                )
            )
            if emit is not None and _changed(run, name, 1):
                passed.append(name)
            continue
        match member:
            case RegistryTool(action=action) if action and action.sensitive_parameters:
                if emit is not None and _changed(run, name, 0):
                    properties = action.input_schema.get("properties", {})
                    key = next(
                        (
                            key
                            for key in sorted(action.sensitive_parameters)
                            if properties.get(key, {}).get("type") == "string"
                        ),
                        None,
                    )
                    reason = f"free str {key!r}" if key is not None else "sensitive parameters"
                    drops.append((name, 0, f"dropped {name}: {reason}"))
            case _ if name == "complete" or has_device_atomic_exception(member):
                choices = await ATOMIC_EXCEPTIONS[name](run)
                options.extend(choices)
                if emit is not None and _changed(run, name, len(choices)):
                    exceptions.append(
                        (
                            name,
                            len(choices),
                            f"{name} → {_exception_summary(name, len(choices))} (exception)",
                        )
                    )
            case _:
                choices, reason = from_schema(run, member)
                options.extend(choices)
                if emit is None or not _changed(run, name, len(choices)):
                    continue
                if not choices:
                    drops.append((name, 0, f"dropped {name}: {reason}"))
                elif not choices[0].parameters:
                    passed.append(name)
                else:
                    schemas.append(
                        (
                            name,
                            len(choices),
                            f"{name} → {schema_summary(choices)} (schema: {reason})",
                        )
                    )
    if emit is not None:
        passthrough = (
            [("passed through", len(passed), f"passed through: {', '.join(passed)}")]
            if passed
            else []
        )
        for key, count, text in exceptions + schemas + passthrough + drops:
            emit(
                ToolDetail(
                    section="tools",
                    key=redact_run_text(key),
                    count=count,
                    text=safe_text(redact_run_text(text), limit=None),
                )
            )
    return tuple(options)

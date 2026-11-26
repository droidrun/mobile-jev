"""Expand finite parameter schemas and report skipped tools."""

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from math import prod

from llama_index.core.tools import BaseTool
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_selector_prompt

from ..helpers import coordinate_pairs


@dataclass(frozen=True)
class AtomicTool:
    name: str
    parameters: dict[str, JSONValue] = dataclass_field(default_factory=dict)
    label: str = ""
    context: str = ""


def _display_value(value: JSONValue) -> str:
    return value if type(value) is str else json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_schema(run: Run, member: BaseTool) -> tuple[list[AtomicTool], str]:
    metadata = member.metadata
    name = metadata.get_name()
    schema = metadata.get_parameters_dict()
    properties = schema.get("properties", {})
    if coordinate_pairs(properties):
        return [], "coordinates"
    domains: dict[str, list[JSONValue]] = {}
    reasons: list[str] = []
    for key, field in sorted(properties.items()):
        ref = field.get("$ref", "")
        if ref.startswith("#/$defs/"):
            field = schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        values: list[JSONValue]
        if name == "press_button" and key == "button":
            values = list(sorted(run.device.supported_buttons))
            kind = "Literal"
        elif "enum" in field:
            values = field["enum"]
            kind = "enum" if ref else "Literal"
        elif "const" in field:
            values = [field["const"]]
            kind = "Literal"
        elif field.get("type") == "boolean":
            values = [False, True]
            kind = "bool"
        else:
            kind = {"integer": "int", "number": "float", "string": "str"}.get(
                field.get("type"), "object"
            )
            return [], "element index" if key == "index" else f"free {kind} {key!r}"
        if not values:
            return [], f"no supported values for {key!r}"
        reasons.append(f"{kind} {key!r}")
        domains[key] = sorted(values, key=lambda value: json.dumps(value, sort_keys=True))
    count = prod(map(len, domains.values()))
    limit = run.tool_config.max_schema_combinations
    if count > limit:
        return [], f"{count} combinations exceed limit {limit}"
    return [
        AtomicTool(
            name,
            dict(zip(domains, values, strict=True)),
            render_selector_prompt(
                run.config.decision.candidate_prompt,
                kind="schema",
                name=name,
                arguments=[_display_value(value) for value in values],
            ),
            metadata.description,
        )
        for values in product(*domains.values())
    ], (reasons[0].split(" ", 1)[0] if len(reasons) == 1 else ", ".join(reasons))


def schema_summary(options: list[AtomicTool]) -> str:
    if len(options[0].parameters) == 1 and len(options) <= 8:
        values = ", ".join(
            _display_value(next(iter(option.parameters.values()))) for option in options
        )
        if len(values) <= 120:
            return values
    return f"{len(options)} combinations"

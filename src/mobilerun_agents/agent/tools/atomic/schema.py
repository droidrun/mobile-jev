"""Expand finite parameter schemas and report skipped tools."""

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from math import prod

from llama_index.core.tools import BaseTool
from mobilerun.core.devices import Device
from mobilerun.models.context import JSONValue

from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig, ToolsConfig

from ..helpers import coordinate_pairs


@dataclass(frozen=True)
class AtomicTool:
    name: str
    parameters: dict[str, JSONValue] = dataclass_field(default_factory=dict)
    label: str = ""
    context: str = ""


def _display_value(value: JSONValue) -> str:
    return value if type(value) is str else json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_schema(
    member: BaseTool, device: Device, tools: ToolsConfig, decision: DecisionConfig
) -> list[AtomicTool]:
    metadata = member.metadata
    name = metadata.get_name()
    schema = metadata.get_parameters_dict()
    properties = schema.get("properties", {})
    if coordinate_pairs(properties):
        return []
    domains: dict[str, list[JSONValue]] = {}
    for key, field in sorted(properties.items()):
        ref = field.get("$ref", "")
        if ref.startswith("#/$defs/"):
            field = schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        values: list[JSONValue]
        if name == "press_button" and key == "button":
            values = list(sorted(device.supported_buttons))
        elif "enum" in field:
            values = field["enum"]
        elif "const" in field:
            values = [field["const"]]
        elif field.get("type") == "boolean":
            values = [False, True]
        else:
            return []
        if not values:
            return []
        domains[key] = sorted(values, key=lambda value: json.dumps(value, sort_keys=True))
    count = prod(map(len, domains.values()))
    limit = tools.max_schema_combinations
    if count > limit:
        return []
    return [
        AtomicTool(
            name,
            dict(zip(domains, values, strict=True)),
            render_prompt(
                decision.candidate_prompt,
                kind="schema",
                name=name,
                arguments=[_display_value(value) for value in values],
            ),
            metadata.description,
        )
        for values in product(*domains.values())
    ]

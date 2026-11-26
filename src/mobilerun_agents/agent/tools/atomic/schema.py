"""Expand finite parameter schemas and report skipped tools."""

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from math import prod

from llama_index.core.tools import BaseTool
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.logs.diagnostics import Detail, sink
from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DecisionConfig, Vision

from ..helpers import coordinate_pairs


@dataclass(frozen=True)
class AtomicTool:
    name: str
    parameters: dict[str, JSONValue] = dataclass_field(default_factory=dict)
    label: str = ""
    context: str = ""


@dataclass(frozen=True)
class Turn:
    run: Run
    config: DecisionConfig
    vision: Vision


def drop(run: Run, name: str, reason: str) -> None:
    emit = sink.get()
    if emit is not None and name not in run.atomic_drops:
        run.atomic_drops.add(name)
        emit(Detail(section="tools", text=f"dropped {name}: {reason}"))


def from_schema(turn: Turn, member: BaseTool) -> list[AtomicTool]:
    metadata = member.metadata
    name = metadata.get_name()
    schema = metadata.get_parameters_dict()
    properties = schema.get("properties", {})
    if coordinate_pairs(properties):
        drop(turn.run, name, "coordinates")
        return []
    domains: dict[str, list[JSONValue]] = {}
    for key, field in sorted(properties.items()):
        if (ref := field.get("$ref", "")).startswith("#/$defs/"):
            field = schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
        values: list[JSONValue]
        if name == "press_button" and key == "button":
            values = list(sorted(turn.run.device.supported_buttons))
        elif "enum" in field:
            values = field["enum"]
        elif "const" in field:
            values = [field["const"]]
        elif field.get("type") == "boolean":
            values = [False, True]
        else:
            kind = {"integer": "int", "number": "float", "string": "str"}.get(
                field.get("type"), "object"
            )
            drop(turn.run, name, "element index" if key == "index" else f"free {kind} {key!r}")
            return []
        if not values:
            drop(turn.run, name, "no supported values")
            return []
        domains[key] = sorted(values, key=lambda value: json.dumps(value, sort_keys=True))
    count = prod(map(len, domains.values()))
    limit = turn.config.max_schema_combinations
    if count > limit:
        drop(turn.run, name, f"{count} combinations exceed limit {limit}")
        return []
    return [
        AtomicTool(
            name,
            dict(zip(domains, values, strict=True)),
            render_selector_prompt(
                turn.config.candidate_prompt,
                kind="schema",
                name=name,
                arguments=[
                    (
                        value
                        if type(value) is str
                        else json.dumps(value, ensure_ascii=False, sort_keys=True)
                    )
                    for value in values
                ],
            ),
            metadata.description,
        )
        for values in product(*domains.values())
    ]

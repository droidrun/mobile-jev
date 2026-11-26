"""Expand a run's registry into fixed actions for choice-based selectors."""

from collections.abc import Mapping

from llama_index.core.tools import BaseTool

from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.user_config import DecisionConfig, Vision

from ..base import RegistryTool
from .exceptions import ATOMIC_EXCEPTIONS
from .schema import AtomicTool, Turn, drop, from_schema

__all__ = ["atomic_tools", "AtomicTool"]


async def atomic_tools(
    members: Mapping[str, BaseTool],
    run: Run,
    vision: Vision = "on",
    config: DecisionConfig | None = None,
) -> tuple[AtomicTool, ...]:
    turn = Turn(run, config or DecisionConfig(), vision)
    options: list[AtomicTool] = []
    for name, member in sorted(members.items()):
        if name == "complete":
            continue  # Completion is a separate output of the decision model.
        if name == "delegate":
            options.append(AtomicTool(name, label=member.metadata.description))
            continue
        match member:
            case RegistryTool(action=action) if action and action.sensitive_parameters:
                properties = action.input_schema.get("properties", {})
                key = next(
                    (
                        key
                        for key in sorted(action.sensitive_parameters)
                        if properties.get(key, {}).get("type") == "string"
                    ),
                    None,
                )
                drop(run, name, f"free str {key!r}" if key else "sensitive parameters")
                continue
            case RegistryTool(action=action) if (
                action and action.owner == "device" and name in ATOMIC_EXCEPTIONS
            ):
                options.extend(await ATOMIC_EXCEPTIONS[name](turn))
            case _:
                options.extend(from_schema(turn, member))
    return tuple(options)

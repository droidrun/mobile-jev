"""Expand the registry into fixed options for decision models."""

from collections.abc import Sequence

from llama_index.core.tools import BaseTool
from mobilerun.core.devices import Device
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.config.user_config import DecisionConfig, ToolsConfig

from ..base import RegistryTool
from .exceptions import ATOMIC_EXCEPTIONS
from .schema import AtomicTool, from_schema

__all__ = ["expand_atomic_tools", "AtomicTool"]


def has_device_atomic_exception(member: BaseTool) -> bool:
    match member:
        case RegistryTool(action=action) if action and action.owner == "device":
            return (
                not action.sensitive_parameters and member.metadata.get_name() in ATOMIC_EXCEPTIONS
            )
    return False


async def expand_atomic_tools(
    members: Sequence[BaseTool],
    device: Device,
    ui: UIState | None,
    goal: str,
    tools: ToolsConfig,
    decision: DecisionConfig,
) -> tuple[AtomicTool, ...]:
    options: list[AtomicTool] = []
    for member in sorted(
        members,
        key=lambda member: (member.metadata.get_name() == "complete", member.metadata.get_name()),
    ):
        name = member.metadata.get_name()
        match member:
            case RegistryTool(action=action) if action and action.sensitive_parameters:
                continue
            case _ if name == "complete" or has_device_atomic_exception(member):
                options.extend(await ATOMIC_EXCEPTIONS[name](device, ui, goal, tools, decision))
            case _:
                options.extend(from_schema(member, device, tools, decision))
    return tuple(options)

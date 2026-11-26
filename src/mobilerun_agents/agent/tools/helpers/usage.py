"""Identify the actions that depend on visual coordinates or UI indices."""

from collections.abc import Iterable
from dataclasses import dataclass

from llama_index.core.tools import BaseTool, ToolMetadata

from .coordinates import coordinate_pairs


@dataclass
class IndexedToolMetadata(ToolMetadata):
    ui_indexed: bool = False


def visual_tool_names(tools: Iterable[BaseTool]) -> tuple[set[str], set[str]]:
    coordinates: set[str] = set()
    indexed: set[str] = set()
    for tool in tools:
        metadata = tool.metadata
        name = metadata.get_name()
        if coordinate_pairs(metadata.get_parameters_dict().get("properties", {})):
            coordinates.add(name)
        match metadata:
            case IndexedToolMetadata(ui_indexed=True):
                indexed.add(name)
    return coordinates, indexed

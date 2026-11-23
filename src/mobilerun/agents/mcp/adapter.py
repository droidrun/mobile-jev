"""Adapter to convert MCP tools to Mobilerun custom tool format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from mobilerun.agents.mcp.client import MCPClientManager


def schema_to_parameters(input_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert JSON Schema properties to Mobilerun parameters format.

    Args:
        input_schema: JSON Schema with "properties" and "required" fields.

    Returns:
        Dict mapping param name to {"type": ..., "required": ..., "default": ...}.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    parameters: dict[str, dict[str, Any]] = {}
    for prop_name, prop_info in properties.items():
        param: dict[str, Any] = {
            "type": prop_info.get("type", "string"),
            "required": prop_name in required,
        }
        if "default" in prop_info:
            param["default"] = prop_info["default"]
        if "description" in prop_info:
            param["description"] = prop_info["description"]
        parameters[prop_name] = param

    return parameters


def mcp_to_mobilerun_tools(
    mcp_manager: "MCPClientManager",
) -> dict[str, dict[str, Any]]:
    """Convert discovered MCP tools to Mobilerun custom tool format."""
    custom_tools: dict[str, dict[str, Any]] = {}

    for tool_name, tool_info in mcp_manager.tools.items():
        wrapper = _create_tool_wrapper(tool_name, mcp_manager)
        custom_tools[tool_name] = {
            "parameters": schema_to_parameters(tool_info.input_schema),
            "description": tool_info.description,
            "function": wrapper,
        }

    return custom_tools


def _create_tool_wrapper(tool_name: str, manager: "MCPClientManager") -> Callable[..., Any]:
    """Create async wrapper function for an MCP tool."""

    async def mcp_tool_wrapper(*, ctx: Any = None, **kwargs: Any) -> str:
        result = await manager.call_tool(tool_name, kwargs)

        if result.content:
            text_parts = []
            for block in result.content:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
            if text_parts:
                return "\n".join(text_parts)

        return str(result)

    mcp_tool_wrapper.__name__ = f"mcp_{tool_name}"
    return mcp_tool_wrapper

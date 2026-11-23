"""ToolRegistry — single source of truth for available tools.

Central registry for agent-callable tools.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
from dataclasses import dataclass
from functools import singledispatch
from html import escape
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional, Set

from mobilerun.logs.actions import agent_line
from mobilerun.logs.actions import logger as action_logger
from mobilerun.logs.handlers import VERBOSE

from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.tools.action_result import ActionResult
from mobilerun.agents.workflows.common.events import ToolExecutionEvent

if TYPE_CHECKING:
    from llama_index.core.workflow import Context as WorkflowContext

    from mobilerun.agents.tools.action_context import ActionContext

logger = logging.getLogger("mobilerun")

_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object"}

# These have agent wrappers for coordinate conversion, recording, or result handling.
# Hide them even when their standard tool is disabled, to avoid bypassing policy.
_WRAPPED_DEVICE_ACTIONS = frozenset(
    {
        "device.tap",
        "device.swipe",
        "device.input_text",
        "device.press_button",
        "device.start_app",
        "device.wait",
    }
)


@singledispatch
def _action_result(result: object) -> ActionResult:
    return ActionResult(success=True, summary=str(result) if result else "Done")


@_action_result.register
def _(result: ActionResult) -> ActionResult:
    return result


@_action_result.register
def _(result: tuple) -> ActionResult:
    return ActionResult(success=result[0], summary=str(result[1]))


@_action_result.register
def _(result: str) -> ActionResult:
    return ActionResult(success=not result.startswith("Failed"), summary=result)


def _json_schema_for_type(type_name: str) -> Dict[str, Any]:
    """Map a registry parameter type to a JSON Schema fragment."""
    if type_name in ("list", "array"):
        return {"type": "array", "items": {}}
    if type_name in _JSON_SCHEMA_TYPES:
        return {"type": type_name}
    # Unknown types (e.g. "any") accept any JSON value.
    return {}


@dataclass
class ToolEntry:
    fn: Callable[..., Awaitable[ActionResult]]
    params: Dict[str, Any]
    description: str
    deps: Optional[Set[str]] = None
    input_schema: Optional[Dict[str, Any]] = None
    sensitive_parameters: frozenset[str] = frozenset()
    device_action: bool = False
    action: Any = None


class ToolRegistry:
    """Central registry of all agent-callable tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, ToolEntry] = {}

    # -- registration --------------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable,
        params: Dict[str, Any],
        description: str,
        deps: Optional[Set[str]] = None,
    ) -> None:
        if name in self.tools and self.tools[name].device_action:
            raise ValueError(f"Duplicate tool name: {name}")

        async def normalized(**kwargs: Any) -> ActionResult:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
            return _action_result(result)

        self.tools[name] = ToolEntry(
            fn=normalized, params=params, description=description, deps=deps
        )

    def register_from_dict(self, tools_dict: Dict[str, Any]) -> None:
        """Register tools from the existing ``{name: {parameters, description, function}}`` format."""
        for name, spec in tools_dict.items():
            deps = spec.get("deps")
            if deps is not None:
                deps = set(deps)
            self.register(
                name=name,
                fn=spec["function"],
                params=spec.get("parameters", {}),
                description=spec.get("description", f"Tool: {name}"),
                deps=deps,
            )

    def register_device_actions(self, device: Any) -> set[str]:
        """Expose the operator catalog without rewriting its JSON schemas."""
        names: set[str] = set()
        for action in device.actions.list(audience="operator"):
            if action.owner == "device":
                properties = action.input_schema.get("properties", {})
                coordinates = any(
                    re.fullmatch(r"(?:x|y|(?:start|end)_[xy]|[xy]\d+)", key) for key in properties
                )
                if action.name in _WRAPPED_DEVICE_ACTIONS or coordinates:
                    continue
            name = action.name.replace(".", "__")
            if len(name) > 64 or re.fullmatch(r"[a-zA-Z0-9_-]+", name) is None:
                name = "action_" + hashlib.sha256(action.name.encode()).hexdigest()[:56]
            if name in self.tools or name in names:
                raise ValueError(f"Duplicate tool name: {name}")

            properties = action.input_schema.get("properties", {})

            async def unused(**kwargs: Any) -> ActionResult:
                raise RuntimeError("Device actions are dispatched by the registry")

            self.tools[name] = ToolEntry(
                fn=unused,
                params=dict(properties),
                description=action.description,
                input_schema=dict(action.input_schema),
                sensitive_parameters=action.sensitive_parameters,
                device_action=True,
                action=action,
            )
            names.add(name)
        return names

    def redact_args(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.tools.get(name)
        sensitive = entry.sensitive_parameters if entry else frozenset()
        return {k: "[REDACTED]" if k in sensitive else v for k, v in args.items()}

    def has_sensitive_actions(self) -> bool:
        return any(entry.sensitive_parameters for entry in self.tools.values())

    def redact_text(self, name: str, args: Dict[str, Any], text: str) -> str:
        entry = self.tools.get(name)
        if entry is None:
            return text
        for key in entry.sensitive_parameters:
            value = args.get(key)
            if value is not None:

                def values(item: Any) -> list[Any]:
                    if isinstance(item, dict):
                        return [
                            item,
                            *(v for child in item.values() for v in values(child)),
                        ]
                    if isinstance(item, list):
                        return [item, *(v for child in item for v in values(child))]
                    return [item] if item is not None else []

                for item in values(value):
                    raw = str(item)
                    encoded = json.dumps(item, ensure_ascii=True)
                    if encoded:
                        text = text.replace(encoded, '"[REDACTED]"')
                    if raw:
                        text = text.replace(escape(raw), "[REDACTED]").replace(raw, "[REDACTED]")
        return text

    def disable(self, tool_names: list[str]) -> None:
        """Remove tools by name (silently ignores unknown names)."""
        for name in tool_names:
            self.tools.pop(name, None)

    def disable_unsupported(self, capabilities: Set[str]) -> None:
        """Remove tools whose ``deps`` set is not satisfied by *capabilities*.

        Tools with ``deps=None`` (custom tools, MCP tools, etc.) are kept.
        """
        to_remove = [
            name
            for name, entry in self.tools.items()
            if entry.deps is not None and not entry.deps <= capabilities
        ]
        if to_remove:
            logger.debug(f"Disabling unsupported tools: {to_remove}")
        self.disable(to_remove)

    # -- query ---------------------------------------------------------------

    def get_signatures(self, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Return ``{name: {parameters, description}}`` for prompt building.

        Args:
            exclude: Optional set of tool names to omit.  Used by
                     ExecutorAgent to hide flow-control tools (e.g.
                     complete) that only FastAgent should see.
        """
        exclude = exclude or set()
        return {
            name: {
                "parameters": entry.params,
                "description": entry.description,
                "input_schema": entry.input_schema,
            }
            for name, entry in self.tools.items()
            if name not in exclude
        }

    # -- execution -----------------------------------------------------------

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        ctx: "ActionContext",
        workflow_ctx: "Optional[WorkflowContext]" = None,
    ) -> ActionResult:
        """Dispatch action by name.

        All actions receive ``ctx=ctx`` as a keyword argument.

        Args:
            workflow_ctx: Optional llama-index workflow Context. When provided,
                a ``ToolExecutionEvent`` is streamed after execution.

        Handles:
        - Unknown tool names
        - Sync vs async functions
        - TypeError (bad arguments)
        - General exceptions
        """
        if name not in self.tools:
            result = ActionResult(
                success=False,
                summary=(f"Unknown tool: {name}. Available: {list(self.tools.keys())}"),
            )
            self._emit_event(workflow_ctx, name, args, result)
            return result

        entry = self.tools[name]
        safe_args = self.redact_args(name, args)
        action = agent_line(ctx, name, safe_args)
        if action is not None:
            action_logger.log(VERBOSE if name == "complete" else logging.INFO, "%s", action)
        try:
            if entry.device_action:
                value = await ctx.driver.actions.call(entry.action.name, args, audience="operator")
                recorder = getattr(ctx, "macro_recorder", None)
                if recorder is not None:
                    try:
                        if entry.action.owner != "device":
                            recorded = {
                                "action_type": "extension_call",
                                "name": entry.action.name,
                                "arguments": safe_args,
                            }
                        else:
                            recorded = {
                                "action_type": entry.action.name.removeprefix("device."),
                                **safe_args,
                            }
                        recorder.record_action(recorded, pre_ui=ctx.ui)
                    except Exception:
                        logger.warning("Could not record device action")
                serialized = json.dumps(entry.action.serialize(value))
                result = ActionResult(
                    success=True, summary=self.redact_text(name, args, serialized)
                )
            else:
                result = await entry.fn(**args, ctx=ctx)
        except TypeError as e:
            result = ActionResult(
                success=False,
                summary=(
                    f"Invalid arguments for {name}"
                    if entry.device_action
                    else f"Invalid arguments for {name}: {e}"
                ),
            )
            self._emit_event(workflow_ctx, name, safe_args, result)
            return result
        except Exception as e:
            result = ActionResult(
                success=False,
                summary=(
                    f"Failed to execute {name}"
                    if entry.device_action
                    else f"Failed to execute {name}: {describe_error(e)}"
                ),
            )
            self._emit_event(workflow_ctx, name, safe_args, result)
            return result

        self._emit_event(workflow_ctx, name, safe_args, result)
        return result

    @staticmethod
    def _emit_event(
        workflow_ctx: "Optional[WorkflowContext]",
        name: str,
        args: Dict[str, Any],
        result: ActionResult,
    ) -> None:
        if workflow_ctx is None:
            return
        workflow_ctx.write_event_to_stream(
            ToolExecutionEvent(
                tool_name=name,
                tool_args=args,
                success=result.success,
                summary=result.summary,
            )
        )

    # -- prompt helpers ------------------------------------------------------

    def get_tool_descriptions_xml(self, exclude: Optional[Set[str]] = None) -> str:
        """Build XML ``<functions>`` block for FastAgent system prompt."""
        exclude = exclude or set()
        lines = ["<functions>"]
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            lines.append(f"<function>{self._spec_to_json(name, entry)}</function>")
        lines.append("</functions>")
        return "\n".join(lines)

    def get_function_tool_specs(self, exclude: Optional[Set[str]] = None) -> list[Dict[str, Any]]:
        """Build OpenAI Responses function tools for native tool calling."""
        exclude = exclude or set()
        specs = []
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            if entry.input_schema is not None:
                specs.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": entry.description,
                        "parameters": entry.input_schema,
                        "strict": False,
                    }
                )
                continue
            properties = {}
            required = []
            for param_name, param_info in entry.params.items():
                schema = _json_schema_for_type(param_info.get("type", "string"))
                if param_info.get("description"):
                    schema["description"] = param_info["description"]
                properties[param_name] = schema
                if param_info.get("required", True):
                    required.append(param_name)
            specs.append(
                {
                    "type": "function",
                    "name": name,
                    "description": entry.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                    "strict": False,
                }
            )
        return specs

    def get_tool_descriptions_text(self, exclude: Optional[Set[str]] = None) -> str:
        """Build text tool descriptions for executor prompts."""
        exclude = exclude or set()
        descriptions = []
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            params = self._format_params(entry.params)
            descriptions.append(f"- {name}({params}): {entry.description}")
        return "\n".join(descriptions)

    def get_param_types(self, exclude: Optional[Set[str]] = None) -> Dict[str, str]:
        """Build a flat ``{param_name: type_string}`` map for XML coercion.

        Note: parameter names are global (not per-tool).
        """
        exclude = exclude or set()
        param_types: Dict[str, str] = {}
        for name, entry in self.tools.items():
            if name in exclude:
                continue
            for param_name, param_info in entry.params.items():
                param_types[param_name] = param_info.get("type", "string")
        return param_types

    def get_action_param_types(self, name: str) -> Dict[str, str]:
        entry = self.tools.get(name)
        if entry is None:
            return {}
        return {key: info.get("type", "string") for key, info in entry.params.items()}

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _format_params(parameters: Dict[str, Any]) -> str:
        parts = []
        for name, info in parameters.items():
            type_str = info.get("type", "string")
            if info.get("required", True):
                parts.append(f"{name}: {type_str}")
            else:
                default = info.get("default", "None")
                parts.append(f"{name}: {type_str} = {default}")
        return ", ".join(parts)

    @staticmethod
    def _spec_to_json(name: str, entry: ToolEntry) -> str:
        if entry.input_schema is not None:
            return json.dumps(
                {
                    "name": name,
                    "description": entry.description,
                    "parameters": entry.input_schema,
                },
                separators=(",", ":"),
            )
        parameters = entry.params
        properties = {}
        required = []
        for param_name, param_info in parameters.items():
            properties[param_name] = {"type": param_info.get("type", "string")}
            if param_info.get("description"):
                properties[param_name]["description"] = param_info["description"]
            if param_info.get("required", True):
                required.append(param_name)
            if "default" in param_info:
                properties[param_name]["default"] = param_info["default"]

        tool_dict = {
            "name": name,
            "description": entry.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        return json.dumps(tool_dict, separators=(",", ":"))

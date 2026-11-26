"""Build the run's tool registry from operator actions and run tools."""

import hashlib
import inspect
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, overload

from llama_index.core.tools import BaseTool
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration
from mobilerun.utils.redaction import secret_scope
from pydantic import BaseModel, ConfigDict, create_model

from mobilerun_agents.agent.logs.diagnostics import safe_text
from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DecisionConfig, Vision
from mobilerun_agents.errors.descriptions import describe_error

from .atomic import AtomicTool, atomic_tools
from .base import ActionMetadata, ActionResult, RegistryTool, Source, Tool, complete, tool
from .helpers import check_point, coordinate_pairs

DEVICE_TOOL_DENYLIST = frozenset(
    {
        "connect",
        "close",
        "bind_extension",
        "unbind_extension",
        "context",
        "screenshot",
        "do",
        "expect",
        "install_apk",
        "install_app",
        "uninstall_app",
        "push_file",
        "pull_file",
        "delete_file",
        "list_files",
        "set_location",
        "reset_location",
        "set_timezone",
        "set_language",
        "press_key_code",
        "get_location",
        "get_clipboard",
    }
)
NATIVE_SENSITIVE_PARAMETERS = {
    "input_text": frozenset({"text"}),
    "set_clipboard": frozenset({"text"}),
}


def model_name(action: Action) -> str:
    name = (
        action.name.removeprefix("device.")
        if action.owner == "device"
        else action.name.replace(".", "__")
    )
    if len(name) > 64 or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        return "action_" + hashlib.sha256(action.name.encode()).hexdigest()[:56]
    return name


def _device_entries(device: Device, state_capabilities: Collection[str]) -> tuple[Action, ...]:
    supported = set(device.supported_operations)
    supported.discard("tap_element")
    if "tap" in supported and "element_index" in state_capabilities:
        supported.add("tap_element")
    entries = {
        action.name: action
        for action in device.actions.list(audience="operator")
        if action.owner != "device"
        or (
            model_name(action) in supported
            and model_name(action) not in DEVICE_TOOL_DENYLIST | {"wait"}
        )
    }
    for name, method in vars(Device).items():
        key = f"device.{name}"
        if (
            name.startswith("_")
            or name not in supported
            or name in DEVICE_TOOL_DENYLIST | {"wait"}
            or key in entries
            or not inspect.iscoroutinefunction(method)
        ):
            continue
        entries[key] = replace(
            Action.compile(
                "device",
                name,
                method,
                Declaration(),
                description=inspect.getdoc(method) or f"Device {name.replace('_', ' ')}",
            ),
            name=name,
        )
    return tuple(
        (
            replace(action, sensitive_parameters=NATIVE_SENSITIVE_PARAMETERS[model_name(action)])
            if action.owner == "device"
            and not action.sensitive_parameters
            and model_name(action) in NATIVE_SENSITIVE_PARAMETERS
            else action
        )
        for action in entries.values()
    )


async def _invoke_device(run: Run, action: Action, arguments: dict[str, Any]) -> ActionResult:
    name = model_name(action)
    masked = {
        key: "[REDACTED]" if key in action.sensitive_parameters else value
        for key, value in arguments.items()
    }
    try:
        validated = action.validate(arguments)
        for x, y in coordinate_pairs(validated):
            if validated[x] is not None and validated[y] is not None:
                try:
                    validated[x], validated[y] = check_point(validated[x], validated[y], run)
                except ValueError as exc:
                    detail = (
                        type(exc).__name__ if action.sensitive_parameters else describe_error(exc)
                    )
                    return ActionResult(False, detail)

        # RecordingDevice owns a separate state view. Refreshing the original
        # view for macro guards cannot retarget a pinned index on this view.
        sensitive = action.sensitive_parameters
        # Base primitive recorders do not honor the extension sensitivity flag.
        # Bypass caller-owned recording facades for sensitive native arguments.
        device = run.device._unwrap() if sensitive and action.owner == "device" else run.device
        view = RecordingDevice(device, state_provider=run.state_provider)
        view._extension_state.observers.remove(view._record_extension)
        view.state._current = run.ui
        recorder = run.macro_recorder
        pre_ui = run.ui
        if recorder is not None and not sensitive:
            try:
                pre_ui = await run.device.state.refresh()
            except Exception:
                pass

        if action.owner == "device":
            method = next(vars(cls)[name] for cls in type(view).__mro__ if name in vars(cls))
            with secret_scope(*(str(validated[key]) for key in sensitive)):
                result = action.validate_result(await method.__get__(view)(**validated))
            if recorder is not None and not sensitive and result is not False:
                for primitive in view.recorded_actions:
                    recorder.record_action(dict(primitive), pre_ui=pre_ui)
        else:
            result = await view.actions.call(action.name, validated, audience="operator")
            if recorder is not None and not sensitive:
                recorder.record_action(
                    {"action_type": "extension_call", "name": action.name, "arguments": arguments},
                    pre_ui=pre_ui,
                )
        shown = "[REDACTED]" if action.sensitive_parameters else repr(result)
        return ActionResult(result is not False, safe_text(f"{name} {masked} → {shown}"))
    except Exception as exc:
        return ActionResult(False, type(exc).__name__)


@tool
async def wait(run: Run) -> ActionResult:
    """Wait for the configured duration."""
    action = run.device.actions.get("device.wait")
    return await _invoke_device(run, action, {"seconds": run.tool_config.wait_duration})


def build_registry(
    run: Run, tools: Sequence[Tool], output_model: type[BaseModel] | None
) -> "ToolRegistry":
    disabled = set(run.tool_config.exclude or ())

    def project(action: Action) -> Action | None:
        name = model_name(action)
        if name not in {"complete", "wait"} and (name in disabled or action.name in disabled):
            return None
        description = action.description
        if action.name == "complete":
            schema = output_model.model_json_schema() if output_model else None
            description = render_selector_prompt(
                "assets/prompts/complete/description.jinja2", output_schema=schema
            )
            properties = dict(action.input_schema.get("properties", {}))
            if schema is None:
                properties.pop("output", None)
            else:
                properties["output"] = {k: v for k, v in schema.items() if k != "$defs"}
            action = replace(
                action,
                input_schema={
                    **action.input_schema,
                    **({"$defs": schema["$defs"]} if schema and "$defs" in schema else {}),
                    "properties": properties,
                },
            )
        return replace(action, description=description)

    run_tools = [member.compile(run) for member in tools]
    if not any(action.name == "complete" for action in run_tools):
        run_tools.append(complete.compile(run))
    run_tools.append(wait.compile(run))
    members: list[BaseTool] = []
    seen: set[str] = set()
    for from_device, actions in (
        (True, _device_entries(run.device, run.state_provider.capabilities)),
        (False, run_tools),
    ):
        for action in actions:
            name = model_name(action)
            if name in seen:
                raise ValueError(f"Duplicate tool name: {name}")
            seen.add(name)
            exposed = project(action)
            if exposed is not None:
                members.append(_route(action, exposed, from_device, output_model, run))
    return ToolRegistry(tuple(members))


def _route(
    action: Action,
    exposed: Action,
    from_device: bool,
    output: type[BaseModel] | None,
    run: Run,
) -> RegistryTool:
    async def invoke(**args: Any) -> object:
        if from_device:
            return await _invoke_device(run, action, args)
        return action.validate_result(await action._call(**action.validate(args)))

    schema = action._arguments
    if not issubclass(schema, BaseModel):
        raise TypeError("Action requires a Pydantic arguments model")
    if exposed.name == "complete":
        fields: dict[str, Any] = {"success": (bool, ...), "message": (str, ...)}
        if output:
            fields["output"] = (output | None, None)
        schema = create_model(
            "CompletionArguments", __config__=ConfigDict(extra="forbid"), **fields
        )
    # Base sanitizes sensitive fields and the run projects its typed completion schema.
    # Keep the actual Pydantic validators; expose exactly the vetted schema to providers.
    if schema.model_json_schema() != exposed.input_schema:
        schema = type(
            schema.__name__,
            (schema,),
            {
                "model_json_schema": classmethod(
                    lambda cls, **kw: deepcopy(dict(exposed.input_schema))
                )
            },
        )
    source = (
        Source.RUN
        if not from_device
        else (Source.DEVICE if action.owner == "device" else f"extension:{action.owner}")
    )
    return RegistryTool(
        ActionMetadata(
            name=model_name(exposed),
            description=exposed.description,
            fn_schema=schema,
            source=source,
            ui_indexed=action.owner == "device" and model_name(exposed) == "tap_element",
        ),
        invoke,
        action,
    )


@dataclass(frozen=True)
class ToolRegistry(Sequence[BaseTool]):
    """Immutable membership; extending or filtering never changes an earlier registry."""

    tools: tuple[BaseTool, ...] = ()
    by_name: Mapping[str, BaseTool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tools = tuple(self.tools)
        names: dict[str, BaseTool] = {}
        for member in tools:
            name = member.metadata.get_name()
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
            names[name] = member
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "by_name", MappingProxyType(names))

    @overload
    def __getitem__(self, index: int) -> BaseTool: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[BaseTool, ...]: ...
    def __getitem__(self, index: int | slice) -> BaseTool | tuple[BaseTool, ...]:
        return self.tools[index]

    def __len__(self) -> int:
        return len(self.tools)

    @property
    def has_sensitive(self) -> bool:
        for t in self.tools:
            match t:
                case RegistryTool(action=action) if action and action.sensitive_parameters:
                    return True
        return False

    async def atomic(
        self, run: Run, vision: Vision = "on", config: DecisionConfig | None = None
    ) -> tuple[AtomicTool, ...]:
        return await atomic_tools(self.by_name, run, vision=vision, config=config)

    def extend(
        self, extra: Iterable[BaseTool], exclude: Iterable[str | Source] = ()
    ) -> "ToolRegistry":
        excluded = frozenset(exclude)

        def keep(member: BaseTool) -> bool:
            source: Source | str = Source.RUN
            match member.metadata:
                case ActionMetadata(source=origin):
                    source = origin
            return member.metadata.get_name() not in excluded and source not in excluded

        return ToolRegistry(tuple(filter(keep, (*self.tools, *extra))))

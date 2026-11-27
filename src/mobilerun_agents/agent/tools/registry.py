"""Build the run's tool registry from operator actions and run tools."""

import hashlib
import inspect
import json
import re
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

from llama_index.core.tools import BaseTool, ToolSelection
from mobilerun.core.devices.base import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration, iter_sensitive_values
from mobilerun.core.runs import recording_scope
from mobilerun.core.state.ui_state import UIState
from mobilerun.utils.redaction import mask_step_args, secret_scope
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from mobilerun_agents.agent.workflows.core.privacy import mask_value
from mobilerun_agents.agent.workflows.core.targets import (
    ActionTarget,
    action_target,
    element_label,
)
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig, ToolsConfig

from .atomic import AtomicTool, expand_atomic_tools
from .base import (
    ActionMetadata,
    ActionResult,
    Completion,
    RegistryTool,
    RunContext,
    Source,
    Tool,
    complete,
    tool,
)
from .helpers import check_point, coordinate_pairs
from .helpers.coordinates import CoordinateError
from .helpers.errors import tool_error_summary

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
# The cloud driver forwards the base float unchanged to this integer wire field.
# Keep the device's model-facing schema; normalize only at the cloud boundary.
_CLOUD_INTEGER_MINIMUMS = {"swipe": {"duration_ms": 10}}
_VERIFICATION_READS = {"mailbox": {"wait_for_code"}, "phone_number": {"wait_for_code", "messages"}}
_ACTION_VERBS = {
    "tap": "tapped",
    "tap_element": "tapped",
    "long_press": "long-pressed",
    "long_press_at": "long-pressed",
    "long_press_element": "long-pressed",
    "input_text": "typed into",
    "input_text_element": "typed into",
    "input_text_on_element": "typed into",
    "type_secret": "typed into",
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
    return tuple(entries.values())


def _cloud_arguments(name: str, validated: dict[str, Any]) -> str | None:
    for parameter, minimum in _CLOUD_INTEGER_MINIMUMS.get(name, {}).items():
        value = validated[parameter]
        if type(value) is float and value.is_integer():
            value = int(value)
        if type(value) is not int:
            return f"CloudValidationError: {parameter} must be an integer"
        if value < minimum:
            return f"CloudValidationError: {parameter} must be at least {minimum}"
        validated[parameter] = value
    return None


@dataclass(frozen=True)
class _VerificationResult:
    summary: str = field(repr=False)


async def _invoke_device(
    device: Device,
    ui: UIState | None,
    action: Action,
    arguments: dict[str, Any],
) -> ActionResult | _VerificationResult:
    name = model_name(action)
    try:
        validated = action.validate(arguments)
        if action.owner == "device" and device._is_cloud:
            if issue := _cloud_arguments(name, validated):
                return ActionResult(False, issue)
        provider = device.state._resolved_provider
        for x, y in coordinate_pairs(validated):
            if validated[x] is not None and validated[y] is not None:
                if ui is None:
                    return ActionResult(False, "Current screen unavailable")
                try:
                    check_point(validated[x], validated[y], ui, provider)
                except CoordinateError as error:
                    return ActionResult(False, f"ValueError: {error}")

        view = DeviceWrapper(device, state_provider=provider)
        view.state._current = ui
        resolved = action_target(action.name, validated, ui)
        cached = _cached_ui(view, device, device._unwrap()) if resolved else ()
        if action.owner == "device":
            if action.sensitive_parameters:
                device._extension_state.sensitive_ui = True
            result = action.validate_result(await getattr(view, name)(**validated))
        else:
            result = await view.actions.call(action.name, validated, audience="operator")
        if resolved is not None and result is not False:
            summary = _target_summary(name, resolved, cached)
        else:
            summary = f"{name} {arguments}" if arguments else name
        if result is not None:
            summary += f" → {result!r}"
        if _verification_read(device, action) and result is not False:
            return _VerificationResult(summary)
        return ActionResult(result is not False, summary)
    except Exception as error:
        return ActionResult(False, tool_error_summary(error))


def _verification_read(device: Device, action: Action) -> bool:
    extension = device.extensions.get(action.owner)
    return extension is not None and action.name.rsplit(".", 1)[-1] in _VERIFICATION_READS.get(
        extension.role or "", set()
    )


def _vault_action(device: Device, action: Action, method: str) -> bool:
    extension = device.extensions.get(action.owner)
    return (
        extension is not None
        and extension.default_name == "vault"
        and action.name.endswith(f".{method}")
    )


def _cached_ui(*devices: Device) -> tuple[tuple[Device, UIState | None], ...]:
    return tuple((device, device.state._current) for device in dict.fromkeys(devices))


def _target_summary(
    name: str,
    target: ActionTarget,
    cached: tuple[tuple[Device, UIState | None], ...],
) -> str:
    verb = _ACTION_VERBS.get(name.rsplit("__", 1)[-1], name.replace("_", " "))
    label = target.label()
    if target.index is not None:
        label = f"[{target.index}] {label}"
    elif target.node is not None:
        label = f"{target.point} {label}"
    summary = f"{verb} {label}"
    for device, previous in cached:
        current = device.state._current
        if current is not None and current is not previous:
            match current.phone_state.get("focusedElement"):
                case dict() as focused if focused:
                    return f"{summary} · focused: {element_label(focused, focus=focused)}"
    return summary


def build_registry(
    device: Device,
    tools: Sequence[Tool],
    tool_config: ToolsConfig,
    output_model: type[BaseModel] | None = None,
    decision_config: DecisionConfig | None = None,
) -> "ToolRegistry":
    disabled = set(tool_config.exclude or ())

    @tool
    async def wait(phone: Device) -> None:
        """Wait for the screen to finish loading or changing."""
        await phone.wait(tool_config.wait_duration)

    def project(action: Action) -> Action | None:
        name = model_name(action)
        if name not in {"complete", "wait"} and (name in disabled or action.name in disabled):
            return None
        description = action.description
        if action.name == "complete":
            schema = output_model.model_json_schema() if output_model else None
            description = render_prompt(
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

    declared = [(member.compile(), member) for member in tools]
    if not any(action.name == "complete" for action, _ in declared):
        declared.append((complete.compile(), complete))
    declared.append((wait.compile(), wait))
    members: list[BaseTool] = []
    seen: set[str] = set()
    device_actions = _device_entries(device, device.state._resolved_provider.capabilities)
    for action, declaration in [*((action, None) for action in device_actions), *declared]:
        name = model_name(action)
        if name in seen:
            raise ValueError(f"Duplicate tool name: {name}")
        seen.add(name)
        exposed = project(action)
        if exposed is not None:
            members.append(_route(action, exposed, declaration, output_model, device))
    return ToolRegistry(device, tool_config, decision_config or DecisionConfig(), tuple(members))


def _route(
    action: Action,
    exposed: Action,
    declaration: Tool | None,
    output: type[BaseModel] | None,
    device: Device,
) -> RegistryTool:
    async def invoke(ctx: RunContext, args: dict[str, Any]) -> object:
        if declaration is None:
            state = await ctx.store.get_state()
            return await _invoke_device(
                device, state.current.ui if state.current else None, action, args
            )
        validated = action.validate(args)
        ui = None
        if "index" in validated or {"x", "y"} <= validated.keys():
            state = await ctx.store.get_state()
            ui = state.current.ui if state.current else None
        targeted = action_target(action.name, validated, ui)
        cached = _cached_ui(device, device._unwrap()) if targeted else ()
        if action.sensitive_parameters:
            device._extension_state.sensitive_ui = True
        result = action.validate_result(await declaration.invoke(ctx, device, validated))
        if targeted is not None:
            summary = _target_summary(model_name(action), targeted, cached)
            match result:
                case None:
                    return ActionResult(True, summary)
                case ActionResult(success=True, summary=detail):
                    return replace(result, summary=f"{summary} → {detail}" if detail else summary)
                case str() as detail:
                    return ActionResult(True, f"{summary} → {detail}" if detail else summary)
        return result

    schema = action._arguments
    if not issubclass(schema, BaseModel):
        raise TypeError("Action requires a Pydantic arguments model")
    if exposed.name == "complete":
        fields: dict[str, Any] = {"success": (bool, ...), "message": (str | None, ...)}
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
        Source.AGENT
        if declaration is not None
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


@dataclass
class _RunPrivacy:
    feedback: list[str] = field(default_factory=list)
    sensitive_seen: bool = False
    # The public known-code redactor does not include registered literal secrets.
    evidence_secrets: set[str] = field(default_factory=set, repr=False)


@dataclass(frozen=True)
class ToolRegistry:
    """Device and custom actions with model-facing selector projections."""

    device: Device = field(repr=False, compare=False)
    config: ToolsConfig = field(default_factory=ToolsConfig)
    decision_config: DecisionConfig = field(default_factory=DecisionConfig)
    _members: tuple[BaseTool, ...] = ()
    by_name: Mapping[str, BaseTool] = field(init=False, repr=False)
    _privacy: _RunPrivacy = field(default_factory=_RunPrivacy, repr=False, compare=False)

    def __post_init__(self) -> None:
        names: dict[str, BaseTool] = {}
        for member in self._members:
            name = member.metadata.get_name()
            if name in names:
                raise ValueError(f"Duplicate tool name: {name}")
            names[name] = member
        object.__setattr__(self, "by_name", MappingProxyType(names))

    def with_tools(self, tools: Sequence[Tool]) -> "ToolRegistry":
        """Expose additional tools for one agent without resetting run privacy."""
        if not tools:
            return self
        disabled = set(self.config.exclude or ())
        seen = set(self.by_name)
        members = list(self._members)
        for member in tools:
            action = member.compile()
            name = model_name(action)
            if name in seen:
                raise ValueError(f"Duplicate tool name: {name}")
            seen.add(name)
            if name not in disabled and action.name not in disabled:
                members.append(_route(action, action, member, None, self.device))
        return replace(self, _members=tuple(members))

    @property
    def has_sensitive(self) -> bool:
        return any(self.is_sensitive_choice(name) for name in self.by_name)

    def is_sensitive_choice(self, name: str) -> bool:
        match self.by_name.get(name):
            case RegistryTool(action=action) if action is not None:
                return (
                    bool(action.sensitive_parameters)
                    or _verification_read(self.device, action)
                    or _vault_action(self.device, action, "type_field")
                )
            case _:
                return False

    @property
    def sensitive_seen(self) -> bool:
        return self._privacy.sensitive_seen

    def tools(
        self, exclude: Collection[str] = (), include: Sequence[BaseTool] = ()
    ) -> tuple[BaseTool, ...]:
        members = dict(self.by_name)
        for member in include:
            members[member.metadata.get_name()] = member
        return tuple(member for name, member in members.items() if name not in exclude)

    async def atomic_tools(
        self,
        exclude: Collection[str] = (),
        ui: UIState | None = None,
        goal: str = "",
        include: Sequence[BaseTool] = (),
    ) -> tuple[AtomicTool, ...]:
        return await expand_atomic_tools(
            self.tools(exclude, include), self.device, ui, goal, self.config, self.decision_config
        )

    def validate(self, call: ToolSelection) -> str | None:
        """Check dispatch arguments without executing the action."""
        name, arguments = call.tool_name, call.tool_kwargs
        match self.by_name.get(name):
            case RegistryTool(action=action) if action is not None:
                # Typed completion output is checked during execution against its own retry budget.
                schema = cast(type[BaseModel], action._arguments)
            case _:
                return f"Unavailable tool: {name}"
        if arguments.keys() - schema.model_fields.keys():
            return f"Unexpected arguments for {name}."
        try:
            schema.model_validate_json(json.dumps(arguments), strict=True)
        except ValidationError as error:
            return f"Invalid arguments for {name}: {tool_error_summary(error, schema)}"
        except (TypeError, ValueError):
            return f"Arguments for {name} must be JSON."
        return None

    def resolve(self, call: ToolSelection) -> ToolSelection:
        """Keep the call id and record the base identity behind its model-facing name."""
        match self.by_name.get(call.tool_name):
            case RegistryTool(action=action) if action is not None:
                if call.tool_name == "wait":
                    return call.model_copy(
                        update={"tool_kwargs": {"seconds": self.config.wait_duration}}
                    )
                return call.model_copy(
                    update={
                        "tool_name": (
                            action.name.removeprefix("device.")
                            if action.owner == "device"
                            else action.name
                        )
                    }
                )
            case _:
                return call

    def _sensitive_parameters(self, call: ToolSelection) -> frozenset[str]:
        match self.by_name.get(call.tool_name):
            case RegistryTool(action=action) if action is not None:
                return action.sensitive_parameters
            case _:
                return frozenset()

    def _call_secrets(self, call: ToolSelection) -> tuple[str, ...]:
        return tuple(
            value
            for key in self._sensitive_parameters(call)
            for value in iter_sensitive_values(call.tool_kwargs.get(key))
        )

    def redact_selection(
        self, calls: list[ToolSelection], thought: str | None
    ) -> tuple[list[ToolSelection], str | None]:
        """Prepare event and memory copies without modifying the calls to be dispatched."""
        secrets = tuple(value for call in calls for value in self._call_secrets(call))
        redact_code = self.device.known_codes_redactor()
        with secret_scope(*self._privacy.evidence_secrets, *secrets):
            safe_calls = []
            for call in calls:
                sensitive = self._sensitive_parameters(call)
                args = mask_step_args(
                    {
                        key: "***" if key in sensitive else value
                        for key, value in call.tool_kwargs.items()
                    },
                    contains_code=lambda text: redact_code(text) != text,
                )
                safe_calls.append(call.model_copy(update={"tool_kwargs": args}))
            return safe_calls, (
                str(mask_value(thought, redact_code)) if thought is not None else None
            )

    def take_verification_feedback(self) -> list[str]:
        """Supply fresh verification results to the next selector, not workflow state."""
        feedback = self._privacy.feedback[:]
        self._privacy.feedback.clear()
        return feedback

    async def execute(self, call: ToolSelection, ctx: RunContext) -> object:
        member = self.by_name.get(call.tool_name)
        match member:
            case RegistryTool(action=action):
                redact_code = self.device.known_codes_redactor()
                secrets = self._call_secrets(call)
                if secrets:
                    self.device.register_sensitive_values(secrets)
                    self._privacy.evidence_secrets.update(secrets)
                with secret_scope(*self._privacy.evidence_secrets):
                    uses_declared_secret = mask_value(call.tool_kwargs) != call.tool_kwargs
                uses_code = any(
                    redact_code(value) != value for value in iter_sensitive_values(call.tool_kwargs)
                )
                if uses_code or self.is_sensitive_choice(call.tool_name):
                    self._privacy.sensitive_seen = True
                keeping = action is None or not (
                    _verification_read(self.device, action)
                    or _vault_action(self.device, action, "list")
                )
                with recording_scope(enabled=keeping):
                    if uses_code and call.tool_name != "complete":
                        self.device._extension_state.sensitive_ui = True
                    result = await member.execute(ctx, call.tool_kwargs)
                    if type(result) is _VerificationResult:
                        self._privacy.feedback.append(result.summary)
                        return ActionResult(
                            True, "Verification details supplied to the next selector."
                        )
                with secret_scope(*self._privacy.evidence_secrets):
                    match result:
                        case Completion():
                            return result  # The workflow keeps public results separate from memory.
                        case ActionResult(summary=summary):
                            return replace(result, summary=str(mask_value(summary, redact_code)))
                        case str():
                            return mask_value(result, redact_code)
                        case _:
                            if (
                                uses_code
                                or uses_declared_secret
                                or self._sensitive_parameters(call)
                            ):
                                return ActionResult(result is not False, call.tool_name)
                            return result
            case _:
                raise ValueError(f"Unknown tool: {call.tool_name}")

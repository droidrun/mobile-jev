"""Declare run tools and adapt validated actions for LlamaIndex."""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum
from functools import wraps
from typing import Any, get_origin, get_type_hints, overload

from llama_index.core.tools import ToolOutput
from llama_index.core.tools.types import AsyncBaseTool
from mobilerun.core.devices import Device
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration
from workflows import Context

from mobilerun_agents.agent.workflows.core.state import AgentState

from .helpers.usage import IndexedToolMetadata


@dataclass(frozen=True)
class Tool:
    function: Callable[..., Awaitable[Any]]
    declaration: Declaration

    injected: dict[str, str]

    async def invoke(
        self, ctx: Context[AgentState], device: Device, arguments: dict[str, Any]
    ) -> Any:
        resources = {"context": ctx, "device": device}
        return await self.function(
            **arguments, **{name: resources[kind] for name, kind in self.injected.items()}
        )

    def compile(self) -> Action:
        fn = self.function

        @wraps(fn)
        async def projected(**kwargs: Any) -> Any:
            return await fn(**kwargs)

        signature = inspect.signature(fn)
        projected.__signature__ = signature.replace(  # type: ignore[attr-defined]
            parameters=[p for p in signature.parameters.values() if p.name not in self.injected]
        )
        entry = Action.compile("agent", fn.__name__, projected, self.declaration)
        return replace(entry, name=fn.__name__)


@overload
def tool(
    fn: Callable[..., Awaitable[Any]], sensitive_parameters: frozenset[str] = frozenset()
) -> Tool: ...


@overload
def tool(
    fn: None = None, sensitive_parameters: frozenset[str] = frozenset()
) -> Callable[[Callable[..., Awaitable[Any]]], Tool]: ...


def tool(
    fn: Callable[..., Awaitable[Any]] | None = None,
    sensitive_parameters: frozenset[str] = frozenset(),
) -> Tool | Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Declare an async tool; Context and Device annotations are injected on execution."""

    def decorate(function: Callable[..., Awaitable[Any]]) -> Tool:
        if not inspect.iscoroutinefunction(function) or not inspect.getdoc(function):
            raise TypeError("Tools must be async functions with a docstring")
        params = inspect.signature(function).parameters.values()
        hints = get_type_hints(function)
        injected = {}
        for p in params:
            if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                raise TypeError("Tool parameters must accept named arguments")
            annotation = hints.get(p.name)
            if annotation is Context or get_origin(annotation) is Context:
                injected[p.name] = "context"
            elif annotation is Device:
                injected[p.name] = "device"
        if any(p.name in sensitive_parameters and p.default is not p.empty for p in params):
            raise TypeError("Sensitive tool parameters cannot declare a default")
        if sensitive_parameters & injected.keys():
            raise TypeError("Injected tool parameters cannot be marked sensitive")
        return Tool(
            function, Declaration(sensitive_parameters=frozenset(sensitive_parameters)), injected
        )

    return decorate if fn is None else decorate(fn)


@dataclass
class ActionResult:
    """What the agent sees after an action runs."""

    success: bool
    summary: str

    def __str__(self) -> str:
        return self.summary


@dataclass(frozen=True)
class Completion:
    success: bool
    message: str | None
    output: dict[str, object] | None


@tool
async def complete(
    success: bool, message: str | None, output: dict[str, object] | None = None
) -> Completion:
    """Mark the task as complete."""
    return Completion(success, message, output)


class Source(Enum):
    DEVICE = "device"
    AGENT = "agent"


@dataclass
class ActionMetadata(IndexedToolMetadata):
    source: Source | str = Source.AGENT


class RegistryTool(AsyncBaseTool):
    """An async LlamaIndex tool that preserves the underlying validated result."""

    def __init__(
        self,
        metadata: ActionMetadata,
        invoke: Callable[[Context[AgentState], dict[str, Any]], Awaitable[object]],
        action: Action | None = None,
    ) -> None:
        self._metadata, self._invoke, self.action = metadata, invoke, action

    @property
    def metadata(self) -> ActionMetadata:
        return replace(self._metadata)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        raise TypeError("Use ToolRegistry.execute with a workflow context")

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        raise TypeError("Use ToolRegistry.execute with a workflow context")

    async def execute(self, ctx: Context[AgentState], args: dict[str, Any]) -> object:
        return await self._invoke(ctx, args)

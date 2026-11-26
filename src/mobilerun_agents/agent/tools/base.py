"""Declare run tools and adapt validated actions for LlamaIndex."""

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum
from types import MethodType
from typing import Any, get_type_hints, overload

from llama_index.core.tools import ToolOutput
from llama_index.core.tools.types import AsyncBaseTool
from mobilerun.core.extensions import Action
from mobilerun.core.extensions.actions import Declaration

from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.prompts import render_prompt

from .helpers.usage import IndexedToolMetadata


@dataclass(frozen=True)
class Tool:
    function: Callable[..., Awaitable[Any]]
    declaration: Declaration

    async def __call__(self, run: Run, *args: Any, **kwargs: Any) -> Any:
        return await self.function(run, *args, **kwargs)

    def compile(self, run: Run) -> Action:
        fn = self.function
        entry = Action.compile("run", fn.__name__, MethodType(fn, run), self.declaration)
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
    """Declare an async tool; its first argument is the current run, never model input."""

    def decorate(function: Callable[..., Awaitable[Any]]) -> Tool:
        if not inspect.iscoroutinefunction(function) or not inspect.getdoc(function):
            raise TypeError("Tools must be async functions with a docstring")
        params = list(inspect.signature(function).parameters.values())
        if (
            not params
            or params[0].name != "run"
            or params[0].kind
            not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            or get_type_hints(function).get("run") is not Run
        ):
            raise TypeError("Tools require run: Run as their first parameter")
        if any(p.name in sensitive_parameters and p.default is not p.empty for p in params):
            raise TypeError("Sensitive tool parameters cannot declare a default")
        return Tool(function, Declaration(sensitive_parameters=frozenset(sensitive_parameters)))

    return decorate if fn is None else decorate(fn)


@dataclass
class ActionResult:
    """What the agent sees after an action runs."""

    success: bool
    summary: str
    output_rejected: bool = False

    def __str__(self) -> str:
        return self.summary


@dataclass(frozen=True)
class Completion:
    success: bool
    message: str
    output: dict[str, object] | None


@tool
async def complete(
    run: Run, success: bool, message: str, output: dict[str, object] | None = None
) -> Completion:
    """Mark the task as complete."""
    return Completion(success, message, output)


@tool
async def delegate(run: Run) -> None:
    """Ask the delegate model to choose the next action when you need its capabilities."""
    raise RuntimeError(render_prompt("assets/prompts/tools/action_result.jinja2"))


class Source(Enum):
    DEVICE = "device"
    RUN = "run"


@dataclass
class ActionMetadata(IndexedToolMetadata):
    source: Source | str = Source.RUN


class RegistryTool(AsyncBaseTool):
    """An async LlamaIndex tool that preserves the underlying validated result."""

    def __init__(
        self,
        metadata: ActionMetadata,
        invoke: Callable[..., Awaitable[object]],
        action: Action | None = None,
    ) -> None:
        self._metadata, self._invoke, self.action = metadata, invoke, action

    @property
    def metadata(self) -> ActionMetadata:
        return replace(self._metadata)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        raise TypeError("Agent tools are async: use acall")

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        if args:
            raise TypeError("Tool arguments must be named")
        result = await self._invoke(**kwargs)
        return ToolOutput(
            tool_name=self.metadata.get_name(),
            raw_input={},
            raw_output=result,
            content=(
                "[REDACTED]" if self.action and self.action.sensitive_parameters else str(result)
            ),
        )

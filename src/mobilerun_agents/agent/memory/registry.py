"""Discover memory block factories without a mutable registration API."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from importlib import metadata
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, runtime_checkable

from llama_index.core.base.llms.types import TextBlock
from llama_index.core.memory import BaseMemoryBlock, StaticMemoryBlock
from pydantic import PositiveInt

from mobilerun_agents.agent.tools.base import Tool
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.config.paths import resolve_prompt_path
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentMemoryConfig, MobileConfig, Section

from .blocks.app_cards import AppCardBlock
from .blocks.goal import GoalBlock
from .blocks.history import HistoryBlock
from .blocks.last_step import LastStepBlock
from .blocks.notes import NotesBlock
from .blocks.subgoals import LastSubgoalBlock, SubgoalHistoryBlock

_Agent = Literal["executor", "planner"]
_GROUP = "mobilerun.memory_blocks"


@dataclass(frozen=True)
class BlockContext:
    """Build-time inputs. Treat config and the optional planner setup state as read-only."""

    agent: _Agent
    platform: str
    options: Mapping[str, object]
    config: MobileConfig
    state: PlanningState | None = None


BlockFactory = Callable[[BlockContext], BaseMemoryBlock[Any]]
NO_MEMORY_BLOCKS: Mapping[str, BlockFactory] = MappingProxyType({})


@runtime_checkable
class _ToolProvider(Protocol):
    def tools(self) -> Sequence[Tool]: ...


def block_tools(blocks: Iterable[BaseMemoryBlock[Any]]) -> list[Tool]:
    """Collect tools from configured blocks, preserving their order."""
    tools: list[Tool] = []
    for block in blocks:
        match block:
            case _ToolProvider():
                tools.extend(block.tools())
    return tools


class _HistoryOptions(Section):
    steps: PositiveInt = 5


class _SubgoalHistoryOptions(Section):
    subgoals: PositiveInt = 5


class _LastStepOptions(Section):
    screenshot: bool = True


def _require_agent(name: str, context: BlockContext, agent: _Agent) -> None:
    if context.agent != agent:
        raise ValueError(
            f"Memory block {name!r} is only available to the {agent}, not the {context.agent}"
        )


def build_setup_block(text: str) -> StaticMemoryBlock:
    """Build the pinned setup block for executor and planner memory."""
    return StaticMemoryBlock(name="setup", static_content=[TextBlock(text=text)], priority=0)


def _setup(context: BlockContext) -> StaticMemoryBlock:
    Section.model_validate(context.options)
    if context.agent == "executor":
        return build_setup_block(render_prompt("assets/prompts/memory/setup_executor.jinja2"))
    if context.state is None:
        raise ValueError("Planner memory block 'setup' requires the planning setup state")
    return build_setup_block(
        render_prompt("assets/prompts/memory/setup_planner.jinja2", executor=context.state.executor)
    )


def _goal(context: BlockContext) -> GoalBlock:
    Section.model_validate(context.options)
    return GoalBlock(include_notes="notes" not in _role_config(context).blocks)


def _last_step(context: BlockContext) -> LastStepBlock:
    _require_agent("last_step", context, "executor")
    options = _LastStepOptions.model_validate(context.options)
    return LastStepBlock(screenshot=options.screenshot)


def _history(context: BlockContext) -> HistoryBlock:
    _require_agent("history", context, "executor")
    options = _HistoryOptions.model_validate(context.options)
    return HistoryBlock(steps=options.steps, config=context.config.agent)


class _UnavailableBlock(StaticMemoryBlock):
    """A built-in that must be omitted on a given platform."""


def _app_cards(context: BlockContext) -> AppCardBlock | _UnavailableBlock:
    _require_agent("app_cards", context, "executor")
    Section.model_validate(context.options)
    if context.platform not in ("android", "ios"):
        return _UnavailableBlock(name="app_cards", static_content=[])
    return AppCardBlock(
        platform=cast(Literal["android", "ios"], context.platform),
        cards_dir=resolve_prompt_path("assets/app_cards"),
    )


def _notes(context: BlockContext) -> NotesBlock:
    Section.model_validate(context.options)
    return NotesBlock()


def _last_subgoal(context: BlockContext) -> LastSubgoalBlock:
    _require_agent("last_subgoal", context, "planner")
    Section.model_validate(context.options)
    return LastSubgoalBlock()


def _subgoal_history(context: BlockContext) -> SubgoalHistoryBlock:
    _require_agent("subgoal_history", context, "planner")
    options = _SubgoalHistoryOptions.model_validate(context.options)
    return SubgoalHistoryBlock(history=options.subgoals)


BUILTIN: Mapping[str, BlockFactory] = MappingProxyType(
    {
        "setup": _setup,
        "goal": _goal,
        "last_step": _last_step,
        "history": _history,
        "app_cards": _app_cards,
        "notes": _notes,
        "last_subgoal": _last_subgoal,
        "subgoal_history": _subgoal_history,
    }
)


@cache
def _installed() -> tuple[Mapping[str, tuple[BlockFactory, str]], str | None]:
    """Cache discovery failures too, so a bad plugin is not imported again on each run."""
    registered: dict[str, tuple[BlockFactory, str]] = {}
    for entry in metadata.entry_points(group=_GROUP):
        name = entry.name
        source = f"entry point {entry.value}"
        if distribution := getattr(entry, "dist", None):
            source += f" ({distribution.name})"
        if name in BUILTIN:
            return {}, f"Memory block {source} reuses the built-in name {name!r}"
        if name in registered:
            return (
                {},
                f"Memory block {name!r} is registered by both {registered[name][1]} and {source}",
            )
        try:
            factory = entry.load()
        except Exception as exc:
            return {}, f"Cannot load memory block {name!r} from {source}: {exc}"
        if not callable(factory):
            return {}, f"Memory block {name!r} from {source} must export a callable factory"
        registered[name] = factory, source
    return MappingProxyType(registered), None


def _source(name: str, installed: Mapping[str, tuple[BlockFactory, str]]) -> str:
    if name in BUILTIN:
        return "built-in"
    if name in installed:
        return installed[name][1]
    return "operator-supplied factory"


def factories(extra: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS) -> Mapping[str, BlockFactory]:
    """Combine built-ins, installed entry points and one operator's factories."""
    installed, error = _installed()
    if error is not None:
        raise ValueError(error)
    combined = {**BUILTIN, **{name: factory for name, (factory, _) in installed.items()}}
    for name, factory in extra.items():
        if name in combined:
            raise ValueError(
                f"Operator-supplied memory block {name!r} conflicts with {_source(name, installed)}"
            )
        combined[name] = factory
    return MappingProxyType(combined)


def build(
    name: str, context: BlockContext, available: Mapping[str, BlockFactory]
) -> BaseMemoryBlock[Any]:
    """Build a named block, verifying its name matches the registration."""
    installed, _ = _installed()
    if name not in available:
        choices = ", ".join(f"{key} ({_source(key, installed)})" for key in sorted(available))
        raise ValueError(f"Unknown memory block {name!r}. Available names: {choices}")
    source = _source(name, installed)
    try:
        block = available[name](context)
    except Exception as exc:
        raise ValueError(f"Cannot build memory block {name!r} from {source}: {exc}") from exc
    match block:
        case BaseMemoryBlock(name=actual) if actual == name:
            return block
        case BaseMemoryBlock(name=actual):
            raise ValueError(
                f"Memory block {name!r} from {source} returned a block named {actual!r}"
            )
        case _:
            raise ValueError(
                f"Memory block {name!r} from {source} did not return a BaseMemoryBlock"
            )


def _role_config(context: BlockContext) -> AgentMemoryConfig:
    return (
        context.config.memory.executor
        if context.agent == "executor"
        else context.config.memory.planner
    )


def configured_blocks(
    context: BlockContext, extra: Mapping[str, BlockFactory]
) -> list[BaseMemoryBlock[Any]]:
    """Build configured blocks in order, omitting unavailable platform-specific blocks."""
    available = factories(extra)
    section = _role_config(context)
    blocks: list[BaseMemoryBlock[Any]] = []
    for name in section.blocks:
        options = MappingProxyType(section.options.get(name, {}))
        match build(name, replace(context, options=options), available):
            case _UnavailableBlock():
                continue
            case block:
                blocks.append(block)
    return blocks

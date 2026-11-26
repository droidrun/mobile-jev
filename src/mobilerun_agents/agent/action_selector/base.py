"""Typed selection contract. Selectors choose actions; they never execute them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.llm import LLM

from mobilerun_agents.agent.action_selector.llm.calls import Usage
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall, ToolCallParseResult
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.base import ActionMetadata
from mobilerun_agents.agent.tools.helpers import coordinate_pairs
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.config.user_config import Vision as Vision

NativeToolAPI = Literal["responses", "chat_completions", "anthropic", "gemini"]


@dataclass
class ActionCandidate:
    action: ToolCall
    description: str
    point_space: tuple[float, float] | None = None
    context: str = ""


@dataclass
class ActionSelection:
    actions: list[ToolCall] = field(default_factory=list)
    abstained: bool = False
    thought: str = ""
    description: str = ""
    memory_update: str = ""
    response: ChatResponse | None = None
    response_text: str = ""
    usage: Usage | None = None
    parse_result: ToolCallParseResult | None = None
    parse_error: str | None = None
    failure: str | None = None
    confidence: float | None = None
    textual_arguments: bool = False


class ActionSelector(ABC):
    debug_label: str = ""

    @property
    def diagnostic_name(self) -> str:
        return self.debug_label or type(self).__name__

    abstention_reason: str = "Action selector abstained without a fallback."

    @property
    def shared_llm(self) -> LLM | None:
        return None

    @abstractmethod
    async def select(self, run: Run) -> ActionSelection: ...


def vision_registry(run: Run, vision: Vision) -> ToolRegistry:
    excluded: set[str] = set()
    for name, member in run.registry.by_name.items():
        fields = member.metadata.get_parameters_dict().get("properties", {})
        if vision == "off" and coordinate_pairs(fields):
            excluded.add(name)
        if vision == "only" or not run.ui or not run.ui.elements:
            match member.metadata:
                case ActionMetadata(ui_indexed=True):
                    excluded.add(name)
    return run.registry.extend((), exclude=excluded)


def sum_usage(*usages: Usage | None) -> Usage | None:
    present = [u for u in usages if u is not None]
    if not present:
        return None

    def total(values: Sequence[int | None]) -> int | None:
        known = [v for v in values if v is not None]
        return sum(known) if len(known) == len(values) else None

    return Usage(
        request_tokens=total([u.request_tokens for u in present]),
        response_tokens=total([u.response_tokens for u in present]),
        total_tokens=total([u.total_tokens for u in present]),
        reasoning_tokens=total([u.reasoning_tokens for u in present]),
        requests=sum(u.requests for u in present),
    )

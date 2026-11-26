"""Typed selection contract. Selectors choose actions; they never execute them."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, field
from hashlib import blake2b
from secrets import token_bytes
from typing import Literal

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms.llm import LLM
from mobilerun.models.context import JSONValue

from mobilerun_agents.agent.action_selector.llm.calls import Usage
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall, ToolCallParseResult
from mobilerun_agents.agent.run import Run
from mobilerun_agents.config.user_config import Vision as Vision

_SIGNATURE_KEY = token_bytes(32)


def action_signature(name: str, args: dict[str, JSONValue]) -> str:
    """Compare actions without storing their (possibly sensitive) arguments."""
    data = json.dumps([name, args], sort_keys=True, separators=(",", ":")).encode()
    return blake2b(data, key=_SIGNATURE_KEY, digest_size=16).hexdigest()


def guarded_turn(run: Run, name: str, signature: str) -> Run:
    turn = copy(run)
    member = run.registry.by_name.get(name)
    # A whole tool can be hidden from the LLM only if it has no arguments.
    if member is not None and not member.metadata.get_parameters_dict().get("properties"):
        turn.selection_exclude |= {name}
    turn.avoid_signature = signature
    return turn


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

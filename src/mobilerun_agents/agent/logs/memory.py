"""Summarize the exact LlamaIndex memory read, without reading any block twice."""

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import Memory
from workflows import Context

from mobilerun_agents.agent.events import Detail
from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState

_log = logging.getLogger("mobilerun_agents.agent.logs.console")
_record_memory = ContextVar("record_agent_memory", default=False)


@asynccontextmanager
async def memory_evidence_scope(enabled: bool) -> AsyncIterator[None]:
    """Emit memory events for an actual recording, even with a quiet console."""
    token = _record_memory.set(enabled)
    try:
        yield
    finally:
        _record_memory.reset(token)


class MemoryDetail(Detail):
    blocks: dict[str, str]
    last_steps: list[str]
    counts: str
    selector: str = "jev"


def _count(value: int) -> str:
    return str(value) if value < 1000 else f"{value / 1000:.1f}".removesuffix(".0") + "k"


def report_memory(
    ctx: Context[AgentState] | Context[PlanningState],
    memory: Memory,
    messages: list[ChatMessage],
    selector: str = "jev",
) -> None:
    if not _record_memory.get() and not _log.isEnabledFor(logging.DEBUG):
        return

    blocks = dict.fromkeys((block.name for block in memory.memory_blocks), "")
    text = "\n".join(
        f"{message.role.value}: {message.content}" for message in messages if message.content
    )
    estimate = memory._estimate_token_count(
        "\n".join(message.content or "" for message in messages) if selector == "jev" else messages
    )
    counts = f"{_count(estimate)}/{_count(memory.token_limit)} tokens"
    lines = [counts]
    for message in messages:
        # The default LlamaIndex template labels the final (possibly truncated)
        # blocks in one text block, whether inserted under the system or user role.
        for block in message.blocks:
            match block:
                case TextBlock(text=rendered) if rendered.startswith(
                    "<memory>\n"
                ) and rendered.endswith("\n</memory>"):
                    for name in blocks:
                        found = re.search(
                            rf"^<{re.escape(name)}>\n(.*?)\n</{re.escape(name)}>$",
                            rendered,
                            re.M | re.S,
                        )
                        if found is not None:
                            blocks[name] = found[1]
    if text:
        lines.append(text)
    ctx.write_event_to_stream(
        MemoryDetail(
            section="memory",
            text="\n".join(lines),
            counts=counts,
            blocks=blocks,
            last_steps=[
                line
                for line in blocks.get("history", "").splitlines()
                if re.match(r"^\d+ · ", line)
            ][-3:],
            selector=selector,
        )
    )

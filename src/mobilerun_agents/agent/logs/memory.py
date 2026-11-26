"""Summarize the exact LlamaIndex memory read, without reading any block twice."""

import json
import logging
import re

from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import Memory
from workflows import Context

from mobilerun_agents.agent.events import Detail
from mobilerun_agents.agent.workflows.core.state import AgentState

_log = logging.getLogger("mobilerun_agents.agent.logs.console")
_STEP = re.compile(r"^(\d+) · (\S+)(?: (\[\d+\]))? ([✓✗])$")


class MemoryDetail(Detail):
    blocks: dict[str, str]
    last_steps: list[str]
    counts: str
    selector: str = "jev"


def _count(value: int) -> str:
    return str(value) if value < 1000 else f"{value / 1000:.1f}".removesuffix(".0") + "k"


def report_memory(
    ctx: Context[AgentState],
    state: AgentState,
    memory: Memory,
    raw: list[ChatMessage],
    sent: list[ChatMessage],
    selector: str = "jev",
) -> None:
    if not _log.isEnabledFor(logging.DEBUG):
        return

    blocks = dict.fromkeys((block.name for block in memory.memory_blocks), "")
    counts = f"{len(sent)} msgs · {_count(memory._estimate_token_count(sent))}/{_count(memory.token_limit)} tokens"
    changes = {record.call.tool_id: record.screen_changed for record in state.actions}
    steps: dict[str, str] = {}
    lines = [counts]
    for source, message in zip(raw, sent, strict=True):
        # The default LlamaIndex template labels the final (possibly truncated)
        # blocks in one text block, whether inserted under the system or user role.
        for block in source.blocks:
            match block:
                case TextBlock(text=text) if text.startswith("<memory>\n") and text.endswith(
                    "\n</memory>"
                ):
                    for name in blocks:
                        found = re.search(
                            rf"^<{re.escape(name)}>\n(.*?)\n</{re.escape(name)}>$",
                            text,
                            re.M | re.S,
                        )
                        if found is not None:
                            blocks[name] = found[1]
        role = message.role.value
        if source.role == "tool":
            first, _, _ = (source.content or "").partition("\n")
            if result := _STEP.fullmatch(first):
                step, name, index, status = result.groups()
                changed = changes.get(str(source.additional_kwargs.get("tool_call_id", "")))
                change = {True: " changed", False: " unchanged", None: ""}[changed]
                steps[step] = f"{step} {name}{index or ''} {status}{change}"
            if message.role != source.role:
                role += " (tool result)"
        if message.content:
            lines.append(f"{role}: {message.content}")
        for block in message.blocks:
            match block:
                case ToolCallBlock(tool_name=name, tool_kwargs=args):
                    lines.append(f"{role}: {name}({json.dumps(args, ensure_ascii=False)})")
    ctx.write_event_to_stream(
        MemoryDetail(
            section="memory",
            text="\n".join(lines),
            counts=counts,
            blocks=blocks,
            last_steps=list(steps.values())[-3:],
            selector=selector,
        )
    )

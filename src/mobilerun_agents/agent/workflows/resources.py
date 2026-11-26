"""Resolve the current invocation's live objects, never its serializable state."""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated

from workflows.resource import Resource

from mobilerun_agents.agent.action_selector.loader import default_selector
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.base import Tool
from mobilerun_agents.agent.tools.registry import build_registry


@dataclass(kw_only=True)
class WorkflowRun(Run):
    pending_tools: list[Tool] | None
    setup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


active_run: ContextVar[WorkflowRun] = ContextVar("agent_run")


async def get_run() -> WorkflowRun:
    run = active_run.get()
    async with run.setup_lock:
        if run.pending_tools is None:
            return run
        if run.action_selector is None:
            run.action_selector = await default_selector("agent")
        run.registry = build_registry(run, run.pending_tools, run.output)
        run.pending_tools = None
    return run


# ResourceManager belongs to the workflow instance, not a run. Resolve on each
# step; the invocation's context variable supplies the same object to its steps.
RunResource = Annotated[WorkflowRun, Resource(get_run, cache=False)]

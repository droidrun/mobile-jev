from collections.abc import Sequence

from workflows.handler import WorkflowHandler

from mobilerun_agents.agent.action_selector import ActionSelector, DelegatingSelector
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock
from mobilerun_agents.agent.memory.memory import build_memory
from mobilerun_agents.agent.tools.base import Tool, delegate
from mobilerun_agents.agent.workflows.agent import AgentWorkflow

from ._run import RunSpec
from .base import _Operator


def start_agent(
    workflow: AgentWorkflow, run: RunSpec, selector: ActionSelector, tools: Sequence[Tool]
) -> WorkflowHandler:
    available = list(tools)
    match selector:
        case DelegatingSelector():
            available.append(delegate)
    return workflow.run(
        device=run.device,
        goal=run.instruction,
        output=run.output,
        max_steps=run.max_steps,
        run_id=run.run_id,
        action_selector=selector,
        tools=available,
        tool_config=run.config.tools,
        config=run.config.agent,
        memory=build_memory(run.config.memory, [DeviceContextBlock()]),
    )


class AgentOperator(_Operator):
    def __init__(self, selector: ActionSelector | None = None, tools: Sequence[Tool] = ()) -> None:
        self._selector = selector
        self._tools = tuple(tools)

    async def _start(self, run: RunSpec) -> WorkflowHandler:
        selector = self._selector
        if selector is None:
            setup = await default_setup()
            selector = setup.selector or setup.executor
            if selector is None:
                raise ValueError("The default setup has no action selector.")
        return start_agent(AgentWorkflow(timeout=run.timeout), run, selector, self._tools)

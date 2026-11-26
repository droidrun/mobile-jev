from collections.abc import Sequence

from workflows.handler import WorkflowHandler

from mobilerun_agents.agent.action_selector import ActionSelector, DelegatingSelector
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.agent.memory.blocks.device_context import DeviceContextBlock
from mobilerun_agents.agent.memory.memory import build_memory
from mobilerun_agents.agent.planner import Planner
from mobilerun_agents.agent.tools.base import Tool, delegate
from mobilerun_agents.agent.workflows.planning import PlanningWorkflow

from ._run import RunSpec
from .base import _Operator


class PlanningOperator(_Operator):
    _budget_field = "planning_max_steps"

    def __init__(
        self,
        planner: Planner | None = None,
        executor: ActionSelector | None = None,
        tools: Sequence[Tool] = (),
    ) -> None:
        self._tools = tuple(tools)
        self._planner = planner
        self._executor = executor

    async def _start(self, run: RunSpec) -> WorkflowHandler:
        planner, executor = self._planner, self._executor
        if planner is None or executor is None:
            setup = await default_setup()
            planner = planner or setup.planner
            executor = executor or setup.executor or setup.selector
        if planner is None or executor is None:
            raise ValueError(
                "Pass a planner and executor or configure a planning setup as default."
            )
        flow = PlanningWorkflow(planner, executor, timeout=run.timeout)
        tools = self._tools
        match executor:
            case DelegatingSelector():
                tools = (*tools, delegate)
        return flow.run(
            device=run.device,
            goal=run.instruction,
            output=run.output,
            max_steps=run.max_steps,
            run_id=run.run_id,
            tools=tools,
            tool_config=run.config.tools,
            config=run.config.agent,
            memory_factory=lambda: build_memory(run.config.memory, [DeviceContextBlock()]),
        )

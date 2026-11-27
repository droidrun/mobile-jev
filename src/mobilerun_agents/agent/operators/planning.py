from collections.abc import Mapping, Sequence
from functools import partial

from mobilerun.core.devices import Device
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult

from mobilerun_agents.agent.action_selector import ActionSelector
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.agent.memory import BlockFactory
from mobilerun_agents.agent.memory.memory import build_planner_memory, prepare_child_memory
from mobilerun_agents.agent.memory.registry import NO_MEMORY_BLOCKS
from mobilerun_agents.agent.planner import Planner
from mobilerun_agents.agent.tools import Tool
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.agent.workflows.planning import PlanningWorkflow, executor_kind
from mobilerun_agents.config.loader import ConfigLoader

from .consumers import T, execute_workflow


class PlanningOperator(Operator):
    def __init__(
        self,
        planner: Planner | None = None,
        executor: ActionSelector | None = None,
        tools: Sequence[Tool] = (),
        memory_blocks: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._tools = tuple(tools)
        self._memory_blocks = dict(memory_blocks)

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        config = ConfigLoader.load()
        planner, executor = self._planner, self._executor
        if planner is None or executor is None:
            setup = await default_setup()
            planner = planner or setup.planner
            executor = executor or setup.executor or setup.selector
        if planner is None or executor is None:
            raise ValueError(
                "Pass a planner and executor or configure a planning setup as default."
            )
        state = PlanningState(executor=executor_kind(executor))
        child_memory_factory = prepare_child_memory(config, device.platform, self._memory_blocks)
        return await execute_workflow(
            partial(
                PlanningWorkflow,
                planner,
                executor,
                config=config.agent,
                output=output,
                child_memory_factory=child_memory_factory,
            ),
            config,
            device,
            instruction,
            self._tools,
            output,
            config.agent.planning_max_steps if max_steps is None else max_steps,
            config.agent.workflow_timeout if timeout is None else timeout,
            memory_factory=lambda _: build_planner_memory(
                config, state, device.platform, extra=self._memory_blocks
            ),
            typed_completion=False,
        )

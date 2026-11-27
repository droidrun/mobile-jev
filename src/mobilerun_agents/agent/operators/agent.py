from collections.abc import Mapping, Sequence
from functools import partial

from mobilerun.core.devices import Device
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult

from mobilerun_agents.agent.action_selector import ActionSelector
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.agent.memory import BlockFactory
from mobilerun_agents.agent.memory.memory import build_executor_memory
from mobilerun_agents.agent.memory.registry import NO_MEMORY_BLOCKS
from mobilerun_agents.agent.tools import Tool
from mobilerun_agents.agent.workflows.agent import AgentWorkflow
from mobilerun_agents.config.loader import ConfigLoader

from .consumers import T, execute_workflow


class AgentOperator(Operator):
    def __init__(
        self,
        selector: ActionSelector | None = None,
        tools: Sequence[Tool] = (),
        memory_blocks: Mapping[str, BlockFactory] = NO_MEMORY_BLOCKS,
        record: bool | None = None,
    ) -> None:
        self._selector = selector
        self._tools = tuple(tools)
        self._memory_blocks = dict(memory_blocks)
        self._record = record

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
        selector = self._selector
        if selector is None:
            setup = await default_setup()
            selector = setup.selector or setup.executor
            if selector is None:
                raise ValueError("The default setup has no action selector.")
        return await execute_workflow(
            partial(AgentWorkflow, selector, config=config.agent, output=output),
            config,
            device,
            instruction,
            self._tools,
            output,
            config.agent.max_steps if max_steps is None else max_steps,
            config.agent.workflow_timeout if timeout is None else timeout,
            memory_factory=lambda: build_executor_memory(
                config, device.platform, extra=self._memory_blocks
            ),
            record=self._record,
        )

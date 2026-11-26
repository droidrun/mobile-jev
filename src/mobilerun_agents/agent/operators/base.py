from abc import abstractmethod
from typing import ClassVar, Literal, TypeVar
from uuid import uuid4

from mobilerun.core.devices import Device
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel
from workflows.handler import WorkflowHandler

from mobilerun_agents.config.loader import ConfigLoader

from ._consumers import RunConsumers
from ._run import RunSpec

T = TypeVar("T", bound=BaseModel)


class _Operator(Operator):
    _budget_field: ClassVar[Literal["max_steps", "planning_max_steps"]] = "max_steps"

    @abstractmethod
    async def _start(self, run: RunSpec) -> WorkflowHandler: ...

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
        default = (
            config.agent.planning_max_steps
            if self._budget_field == "planning_max_steps"
            else config.agent.max_steps
        )
        budget = default if max_steps is None else max_steps
        deadline = config.agent.workflow_timeout if timeout is None else timeout
        run = RunSpec(device, instruction, output, budget, deadline, uuid4().hex, config)
        async with RunConsumers(config, run.run_id, run.max_steps) as consumers:
            result, path = await consumers.consume(await self._start(run))
            value = result.output
            return ExecutionResult(
                success=result.success,
                answer=result.answer or result.reason,
                output=output.model_validate(value) if output and value is not None else None,
                steps=result.steps,
                device=device.identifier or device.platform,
                trajectory_path=path,
            )

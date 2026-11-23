"""Reasoning-enabled entry point for the existing agent operator."""

from copy import deepcopy
from typing import TypeVar

from mobilerun.core.devices.base import Device
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun.agents.config.loader import ConfigLoader
from mobilerun.agents.operators.agent import AgentOperator

T = TypeVar("T", bound=BaseModel)


class DelegationOperator(AgentOperator):
    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        config = deepcopy(self.config if self.config is not None else ConfigLoader.load())
        config.agent.reasoning = True
        operator = AgentOperator(
            config=config, llms=self.llms, context_provider=self.context_provider
        )
        return await operator.execute(
            device, instruction, output=output, max_steps=max_steps, timeout=timeout
        )

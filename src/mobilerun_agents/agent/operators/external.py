"""Run an operator registered by an installed package."""

from functools import cached_property
from importlib import metadata
from inspect import isclass
from typing import TypeVar

from mobilerun.core.devices import Device
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel, InstanceOf, TypeAdapter, ValidationError

T = TypeVar("T", bound=BaseModel)


class ExternalOperator(Operator):
    def __init__(self, plugin: str) -> None:
        self._plugin = plugin

    @cached_property
    def _operator(self) -> Operator:
        entries = list(metadata.entry_points(group="mobilerun.agents", name=self._plugin))
        if len(entries) != 1:
            raise ValueError(
                f"Expected one installed operator entry point named {self._plugin!r} "
                f"in 'mobilerun.agents', found {len(entries)}"
            )
        exported: object = entries[0].load()
        try:
            if not isclass(exported):
                return TypeAdapter(InstanceOf[Operator]).validate_python(exported)
            operator_class = TypeAdapter(type[Operator]).validate_python(exported)
        except ValidationError:
            raise ValueError(
                f"External plugin {self._plugin!r} must export a mobilerun.core.operator.Operator "
                "instance or a no-argument subclass"
            ) from None
        return operator_class()

    def check(self, device: Device) -> None:
        self._operator.check(device)

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        return await self._operator.execute(
            device, instruction, output=output, max_steps=max_steps, timeout=timeout
        )

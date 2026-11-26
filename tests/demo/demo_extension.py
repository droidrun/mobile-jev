"""Give the agent a tool that reads a local inventory label."""

import asyncio

from mobilerun import AndroidDevice
from mobilerun.core.extensions import Extension, action
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun_agents import AgentOperator

# --- edit me ---
DEVICE_ID = "emulator-5554"
LABEL = "QA bench 3"
GOAL = "Read the inventory label with the bench tool and report it."


class Bench(Extension):
    default_name = "bench"

    @action
    async def inventory_label(self) -> str:
        """Read the inventory label for this bench."""
        return LABEL


async def main() -> None:
    async with AndroidDevice(DEVICE_ID) as device:
        bench = await Bench().bind(device)
        try:
            result: ExecutionResult[BaseModel] = await AgentOperator().execute(device, GOAL)
            print(result.success, result.answer)
        finally:
            await bench.unbind()


if __name__ == "__main__":
    asyncio.run(main())

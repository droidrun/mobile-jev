from dataclasses import dataclass

from mobilerun.core.devices import Device
from pydantic import BaseModel

from mobilerun_agents.config.user_config import MobileConfig


@dataclass(frozen=True)
class RunSpec:
    device: Device
    instruction: str
    output: type[BaseModel] | None
    max_steps: int
    timeout: float | None
    run_id: str
    config: MobileConfig

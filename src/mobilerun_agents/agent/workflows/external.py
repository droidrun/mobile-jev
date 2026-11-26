"""Installed ``mobilerun.agents`` entry points, without in-package discovery."""

import inspect
from collections.abc import Awaitable, Callable
from importlib import metadata
from types import ModuleType
from typing import Any

from mobilerun.core.devices import Device
from pydantic import BaseModel, ConfigDict, TypeAdapter
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.config.settings import DEFAULT_MAX_STEPS

from .core.result import AgentResult
from .core.state import AgentState


class Plugin(BaseModel):
    run: Callable[..., Awaitable[dict[str, object]]]
    config: dict[str, object]


def load_plugin(name: str) -> Plugin:
    entries = list(metadata.entry_points(group="mobilerun.agents", name=name))
    if len(entries) != 1:
        raise ValueError(
            f"Expected one installed external agent named {name!r}, found {len(entries)}"
        )
    module = TypeAdapter(
        ModuleType, config=ConfigDict(arbitrary_types_allowed=True)
    ).validate_python(entries[0].load())
    plugin = Plugin(run=vars(module)["run"], config=vars(module).get("DEFAULT_CONFIG", {}))
    if not inspect.iscoroutinefunction(plugin.run):
        raise ValueError(f"External agent {name!r} must export an async run function")
    return plugin


class ExternalStart(StartEvent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: Device
    goal: str
    output: type[BaseModel] | None = None
    config: dict[str, object] = {}
    max_steps: int = DEFAULT_MAX_STEPS


class ExternalWorkflow(Workflow):
    def __init__(self, name: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.name = name

    @step
    async def execute(self, ctx: Context[AgentState], ev: ExternalStart) -> StopEvent:
        state = AgentState(goal=ev.goal, max_steps=ev.max_steps)
        await ctx.store.set_state(state)
        plugin = load_plugin(self.name)
        raw = await plugin.run(
            device=ev.device,
            instruction=ev.goal,
            config=plugin.config | ev.config,
            max_steps=state.max_steps,
            output_schema=ev.output.model_json_schema() if ev.output else None,
        )
        result = AgentResult[Any].model_validate(raw)
        if ev.output and (result.success or result.output is not None):
            result.output = ev.output.model_validate(result.output)
        return StopEvent(result=result)

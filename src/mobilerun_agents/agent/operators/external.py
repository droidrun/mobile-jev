from workflows.handler import WorkflowHandler

from mobilerun_agents.agent.workflows.external import ExternalWorkflow

from ._run import RunSpec
from .base import _Operator


class ExternalOperator(_Operator):
    def __init__(self, plugin: str) -> None:
        self._plugin = plugin

    async def _start(self, run: RunSpec) -> WorkflowHandler:
        return ExternalWorkflow(self._plugin, timeout=run.timeout).run(
            device=run.device,
            goal=run.instruction,
            output=run.output,
            max_steps=run.max_steps,
            run_id=run.run_id,
        )

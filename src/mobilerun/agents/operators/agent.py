"""Built-in LLM-backed device operator."""

import asyncio
import logging
from copy import deepcopy
from typing import TypeVar, cast

from llama_index.core.llms.llm import LLM
from mobilerun.core.devices.base import Device
from mobilerun.core.operator import Operator
from mobilerun.logs.actions import action_counting
from mobilerun.logs.handlers import apply_log_level, ensure_agent_logging
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun.agents.config.loader import ConfigLoader
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.context.context_provider import ContextProvider
from mobilerun.agents.errors import AgentTimeout
from mobilerun.agents.llm.llm_loader import load_agent_llms
from mobilerun.agents.workflows.droid.droid_agent import MobileAgent
from mobilerun.agents.workflows.droid.events import ResultEvent

_agent_hint_logged = False
T = TypeVar("T", bound=BaseModel)


class AgentOperator(Operator):
    def __init__(
        self,
        config: MobileConfig | None = None,
        llms: LLM | dict[str, LLM] | None = None,
        context_provider: ContextProvider | None = None,
    ) -> None:
        self.config = config
        self.llms = llms
        self.context_provider = context_provider

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        ensure_agent_logging()
        _log_agent_hint()
        config = deepcopy(self.config if self.config is not None else ConfigLoader.load())
        apply_log_level(config.logging)
        if max_steps is not None:
            config.agent.max_steps = max_steps
        llms = (
            self.llms
            if self.llms is not None
            or getattr(config.agent, "name", "mobilerun") not in {"mobilerun", "droidrun"}
            else load_agent_llms(config=config, output_model=output)
        )
        agent = MobileAgent(
            goal=instruction,
            config=config,
            llms=llms,
            driver=device,
            output_model=output,
            context_provider=self.context_provider,
        )
        deadline = asyncio.timeout(timeout)
        try:
            with action_counting():
                async with deadline:
                    result: ResultEvent = await agent.run()
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise AgentTimeout(f"Agent operation timed out after {timeout} seconds") from exc
        structured = cast(T | None, result.structured_output)
        answer = getattr(result, "answer", None)
        if answer is None:
            answer = result.reason
        success = result.success
        if output is not None and result.structured_output is not None:
            try:
                structured = output.model_validate(result.structured_output)
            except Exception as exc:
                success = False
                answer = str(exc)
                structured = None
        trajectory_path = agent.trajectory.trajectory_folder if agent.trajectory else None
        return ExecutionResult[T](
            success=success,
            answer=answer,
            output=structured,
            steps=result.steps,
            trajectory_path=(str(trajectory_path) if trajectory_path is not None else None),
            device=type(device).__name__,
        )


def _log_agent_hint() -> None:
    global _agent_hint_logged
    if not _agent_hint_logged:
        logging.getLogger("mobilerun").debug(
            "device.do()/expect() run the built-in Mobilerun agent. Install the agent with pip install mobilerun-agents."
        )
        _agent_hint_logged = True

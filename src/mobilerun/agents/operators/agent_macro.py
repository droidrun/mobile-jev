"""Guarded replay with an agent taking over on state divergence."""

import asyncio
import json
from pathlib import Path
from typing import TypeVar, Unpack

from llama_index.core.llms.llm import LLM
from mobilerun.config.prompts import PromptLoader
from mobilerun.core.devices.base import Device
from mobilerun.core.operator import Operator
from mobilerun.macro import MacroMismatch, MacroOperator
from mobilerun.macro.operator import PlayerOptions
from mobilerun.macro.player import MacroPlayer
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.operators.agent import AgentOperator

PROMPTS = Path(__file__).resolve().parent.parent / "assets" / "prompts" / "agent_macro"
T = TypeVar("T", bound=BaseModel)


class MacroOptions(PlayerOptions, total=False):
    start_from_step: int


class AgentMacroOperator(MacroOperator):
    def __init__(
        self,
        macro_file: str | None = None,
        *,
        agent: Operator | None = None,
        config: MobileConfig | None = None,
        llms: LLM | dict[str, LLM] | None = None,
        **options: Unpack[MacroOptions],
    ) -> None:
        super().__init__(macro_file, **options)
        self.agent = agent if agent is not None else AgentOperator(config=config, llms=llms)

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        async with asyncio.timeout(timeout):
            return await self._execute(
                device, instruction, output=output, max_steps=max_steps, timeout=timeout
            )

    async def _execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        player = MacroPlayer(device, **self.player_options)
        data = player.load_macro_from_file(self.macro_file or instruction)
        goal = (
            instruction if self.macro_file else data.get("description", "Finish the recorded task")
        )
        try:
            result = await self._execute_macro(player, device, data, max_steps=max_steps)
        except MacroMismatch as mismatch:
            prompt = await PromptLoader.load_prompt(
                str(PROMPTS / "diverged.jinja2"),
                {
                    "goal": goal,
                    "step": mismatch.step_index + 1,
                    "actions": json.dumps(mismatch.remaining_actions),
                    "state": json.dumps(mismatch.current_state),
                },
            )
            remaining_budget = (
                None if max_steps is None else max(0, max_steps - player.executed_steps)
            )
            agent_result = await self.agent.execute(
                device,
                prompt,
                output=output,
                max_steps=remaining_budget,
                **({"timeout": timeout} if timeout is not None else {}),
            )
            return agent_result.model_copy(
                update={"steps": player.executed_steps + (agent_result.steps or 0)}
            )
        if output is not None and result.success:
            prompt = await PromptLoader.load_prompt(str(PROMPTS / "extract.jinja2"), {"goal": goal})
            remaining_budget = (
                None if max_steps is None else max(0, max_steps - player.executed_steps)
            )
            agent_result = await self.agent.execute(
                device,
                prompt,
                output=output,
                max_steps=remaining_budget,
                **({"timeout": timeout} if timeout is not None else {}),
            )
            return agent_result.model_copy(
                update={"steps": player.executed_steps + (agent_result.steps or 0)}
            )
        return result

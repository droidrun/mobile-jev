"""Guarded macro replay; only a pre-state mismatch triggers recovery."""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import TypeVar, Unpack

from mobilerun.core.devices.base import Device
from mobilerun.core.operator import Operator
from mobilerun.macro.errors import MacroMismatch
from mobilerun.macro.operator import MacroOperator, PlayerOptions
from mobilerun.macro.player import MacroPlayer
from mobilerun.models.results import ExecutionResult
from mobilerun.utils.prompts import PromptLoader
from pydantic import BaseModel

from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import MobileConfig

from .paths import resolve_macro_target

_PROMPTS = Path(__file__).resolve().parents[1] / "assets/prompts/agent_macro"
T = TypeVar("T", bound=BaseModel)


class MacroAgentOperator(MacroOperator):
    """Pass a macro path to execute, or set macro_file and pass a goal instead."""

    def __init__(
        self,
        macro_file: str | Path | None = None,
        agent: Operator | None = None,
        config: MobileConfig | None = None,
        start_from_step: int = 0,
        **player_options: Unpack[PlayerOptions],
    ) -> None:
        super().__init__(macro_file, start_from_step=start_from_step, **player_options)
        self._agent = agent
        self._config = config

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        async with asyncio.timeout(timeout) as deadline:
            player = MacroPlayer(device, **self.player_options)
            target = resolve_macro_target(self.macro_file or instruction)
            data = player.load_macro_from_file(str(target))
            variables: dict[str, object] = {}
            try:
                replay = await self._execute_macro(player, device, data, max_steps=max_steps)
            except MacroMismatch as mismatch:
                template = "diverged.jinja2"
                variables.update(
                    step=mismatch.step_index + 1,
                    actions=json.dumps(mismatch.remaining_actions),
                    state=json.dumps(mismatch.current_state),
                )
            else:
                if output is None or not replay.success:
                    return replay
                template = "extract.jinja2"

            variables["goal"] = (
                (instruction if self.macro_file else None)
                or data.get("description")
                or await PromptLoader.load_prompt(str(_PROMPTS / "fallback_goal.jinja2"))
            )
            prompt = await PromptLoader.load_prompt(str(_PROMPTS / template), variables)
            agent = self._agent
            if agent is None:
                from mobilerun_agents.agent.operators import AgentOperator

                agent = AgentOperator()
            end = deadline.when()
            with (
                ConfigLoader.override(self._config)
                if self._agent is None and self._config is not None
                else nullcontext()
            ):
                result = await agent.execute(
                    device,
                    prompt,
                    output=output,
                    max_steps=None if max_steps is None else max_steps - player.executed_steps,
                    timeout=(
                        None if end is None else max(0.0, end - asyncio.get_running_loop().time())
                    ),
                )
            return result.model_copy(update={"steps": player.executed_steps + (result.steps or 0)})

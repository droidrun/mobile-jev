"""Edit the block below, then run `python tests/demo/demo.py`."""

import asyncio
import os
from asyncio import sleep
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import monotonic
from typing import TypeVar

import click
from llama_index.core.llms import ChatMessage
from mobilerun import AndroidDevice, CloudDevice, Device, RemoteDevice
from mobilerun.core.operator import Operator
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.macro.settings import get_macro_settings
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel, Field

from mobilerun_agents import (
    AgentOperator,
    DecisionSelector,
    DelegatingSelector,
    LlamaIndexLLMSelector,
    LlamaIndexPlanner,
    MacroAgentOperator,
    PlanningOperator,
    Run,
    tool,
)
from mobilerun_agents.agent.action_selector import Vision
from mobilerun_agents.agent.action_selector.decision_model import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.factory import build_llm
from mobilerun_agents.agent.auth import CredentialMissing
from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.logs.diagnostics import private_values, redact
from mobilerun_agents.config.loader import ConfigLoader

# --- edit me ---------------------------------------------------------
DEVICE = "86f8d29f-9f59-4e42-ba02-1043c721774e"  # serial, remote URL, or cloud ID
CLOUD = True
RUN = ["all"]  # names in DEMOS below, or ["all"]
SETUPS = ["jev_only", "no_vision", "vision"]  # names in SETUP_DEMOS, or ["all"]
RESET = True  # press Home before each demo and comparison run
DEBUG = True  # True gives the CLI's per-step --debug detail
GOAL = "Open Settings and check the battery status"
MAX_STEPS = 15
TIMEOUT = 300  # seconds per operator run; None = no deadline
MODELS = {
    "openai": "gpt-5.6-terra",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}
COMPARE = ["openai", "anthropic"]
DEFAULT = "openai"
VISION: Vision = "on"  # default for LLM demos without an explicit mode
VISION_MODEL = ("openai", "gpt-5.6-terra")  # provider/model for explicit LLM vision modes
PLANNER = ("openai", "gpt-5.6-terra")
EXECUTOR = ("openai", "gpt-5.6-terra")
JEV_MIN_CONFIDENCE = 0.7
MACRO: str | Path | None = None  # newest recorded macro, or specify a folder/file
# ---------------------------------------------------------------------


class Battery(BaseModel):
    level: int = Field(ge=0, le=100)
    charging: bool


T = TypeVar("T", bound=BaseModel)


@tool
async def note(run: Run, text: str) -> str:
    """Remember an observation for the rest of this run."""
    await run.remember(ChatMessage(role="user", content=text))
    return "Note saved."


async def selector(
    provider: str, model: str | None = None, vision: Vision | None = None
) -> LlamaIndexLLMSelector:
    return LlamaIndexLLMSelector(
        await build_llm(provider, model or MODELS[provider]), vision=vision or VISION
    )


async def execute(
    operator: Operator, phone: Device, goal: str, output: type[BaseModel] | None = None
) -> ExecutionResult[BaseModel]:
    # Keep the demo's redaction context outside the normal (non-debug) workflow.
    token = private_values.set(None) if not DEBUG else None
    try:
        return await operator.execute(
            phone, goal, output=output, max_steps=MAX_STEPS, timeout=TIMEOUT
        )
    finally:
        if token is not None:
            private_values.reset(token)


def summary(result: ExecutionResult[T]) -> str:
    return f"success={result.success} steps={result.steps} answer={result.answer or '-'}"


def line(value: str) -> str:
    return " ".join(redact(value).split())


async def reset_phone(phone: Device) -> None:
    if RESET:
        await phone.press_button("home")
        await sleep(0.5)


async def basic(phone: Device) -> ExecutionResult[BaseModel]:
    agent = AgentOperator(selector=await selector(DEFAULT))
    return await execute(agent, phone, GOAL)


async def compare_models(phone: Device) -> str:
    rows = ["provider/model\tsuccess\tsteps\tseconds\tanswer"]
    for provider in COMPARE:
        start = monotonic()
        success: bool | str
        try:
            agent = AgentOperator(selector=await selector(provider))
            await reset_phone(phone)
            result = await execute(agent, phone, GOAL)
            success, steps, answer = result.success, result.steps, result.answer or "-"
        except CredentialMissing:
            success, steps, answer = "-", None, "skipped (no key)"
        except Exception as error:
            success, steps, answer = False, None, f"{type(error).__name__}: {error}"
        model = MODELS.get(provider, "?")
        rows.append(
            f"{line(f'{provider}/{model}')}\t{success}\t{steps if steps is not None else '-'}\t"
            f"{monotonic() - start:.1f}\t{line(answer)}"
        )
    print("\n".join(rows))
    return f"Compared {len(COMPARE)} model(s)."


def jev(mode: Vision | None = None) -> DecisionSelector:
    model = TypeSafeDecisionModel()
    return DecisionSelector(model) if mode is None else DecisionSelector(model, vision=mode)


def jev_without_buttons() -> DecisionSelector:
    return DecisionSelector(TypeSafeDecisionModel(), exclude={"press_button"})


async def jev_only(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(AgentOperator(selector=jev()), phone, GOAL)


async def jev_exclude(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(AgentOperator(selector=jev_without_buttons()), phone, GOAL)


async def delegated_selector() -> DelegatingSelector:
    return DelegatingSelector(
        jev("off"),
        await selector(*VISION_MODEL, vision="on"),
        min_confidence=JEV_MIN_CONFIDENCE,
    )


async def jev_delegate(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(AgentOperator(selector=await delegated_selector()), phone, GOAL)


async def jev_executor(phone: Device) -> ExecutionResult[BaseModel]:
    operator = PlanningOperator(
        planner=LlamaIndexPlanner(await build_llm(*PLANNER)), executor=await delegated_selector()
    )
    return await execute(operator, phone, GOAL)


async def no_vision(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(
        AgentOperator(selector=await selector(*VISION_MODEL, vision="off")), phone, GOAL
    )


async def vision(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(
        AgentOperator(selector=await selector(*VISION_MODEL, vision="on")), phone, GOAL
    )


async def vision_only(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(
        AgentOperator(selector=await selector(*VISION_MODEL, vision="only")), phone, GOAL
    )


async def jev_no_vision(phone: Device) -> ExecutionResult[BaseModel]:
    operator = AgentOperator(selector=jev("off"))
    return await execute(operator, phone, GOAL)


async def jev_vision(phone: Device) -> ExecutionResult[BaseModel]:
    operator = AgentOperator(selector=jev("on"))
    return await execute(operator, phone, GOAL)


async def jev_vision_only(phone: Device) -> ExecutionResult[BaseModel]:
    operator = AgentOperator(selector=jev("only"))
    return await execute(operator, phone, GOAL)


async def compare_setups(phone: Device) -> str:
    rows = ["setup\tsuccess\tsteps\tseconds\ttokens\tanswer"]
    names = list(SETUP_DEMOS) if SETUPS == ["all"] else SETUPS
    for name in names:
        start = monotonic()
        success, steps, answer = False, None, "Unknown setup"
        if setup := SETUP_DEMOS.get(name):
            try:
                await reset_phone(phone)
                result = await setup(phone)
                success, steps, answer = result.success, result.steps, result.answer or "-"
            except Exception as error:
                answer = f"{type(error).__name__}: {error}"
        # The public ExecutionResult does not report model token usage.
        rows.append(
            f"{line(name)}\t{success}\t{steps if steps is not None else '-'}\t"
            f"{monotonic() - start:.1f}\t-\t{line(answer)}"
        )
    print("\n".join(rows))
    return f"Compared {len(names)} setup(s)."


async def planning(phone: Device) -> ExecutionResult[BaseModel]:
    operator = PlanningOperator(
        planner=LlamaIndexPlanner(await build_llm(*PLANNER)), executor=await selector(*EXECUTOR)
    )
    return await execute(operator, phone, GOAL)


async def structured(phone: Device) -> str:
    agent = AgentOperator(selector=await selector(DEFAULT))
    result = await execute(agent, phone, GOAL, output=Battery)
    return result.output.model_dump_json() if result.output else summary(result)


async def custom_tool(phone: Device) -> str:
    agent = AgentOperator(selector=await selector(DEFAULT), tools=[note])
    return summary(await execute(agent, phone, GOAL + ". Save the observation with note."))


async def macro(phone: Device) -> str:
    macro_file = MACRO
    if macro_file is None:
        directory = Path(get_macro_settings().macro_directory).expanduser()
        macro_file = max(
            (path for path in directory.glob("*/macro.json") if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path),
            default=None,
        )
        if macro_file is None:
            return "skipped: no recorded macro yet (run any demo first)"
    fallback = AgentOperator(selector=await selector(DEFAULT))
    return summary(
        await execute(MacroAgentOperator(macro_file=macro_file, agent=fallback), phone, GOAL)
    )


async def device(phone: Device) -> str:
    screenshot = await phone.screenshot()
    state = await phone.state.refresh()
    try:
        inventory = f"{len(await phone.list_apps())} apps"
    except DeviceOperationNotSupported:
        inventory = "app list unavailable"
    return f"Screenshot: {len(screenshot)} bytes; {len(state.elements)} UI elements; {inventory}"


SETUP_DEMOS = {
    fn.__name__: fn
    for fn in (
        basic,
        jev_only,
        jev_exclude,
        jev_delegate,
        jev_executor,
        no_vision,
        vision,
        vision_only,
        jev_no_vision,
        jev_vision,
        jev_vision_only,
        planning,
    )
}

DEMOS: dict[str, Callable[[Device], Awaitable[str | ExecutionResult[BaseModel]]]] = {
    **SETUP_DEMOS,
    **{
        fn.__name__: fn
        for fn in (compare_models, compare_setups, structured, custom_tool, macro, device)
    },
}


async def main() -> None:
    with click.Context(click.Command("demo")):
        configure_cli_logging(debug=DEBUG)
        await _run_demos()


async def _run_demos() -> None:
    # The factory registers saved provider keys for redaction in this context too.
    token = private_values.set(
        {
            v
            for k, v in os.environ.items()
            if v and k.endswith(("_KEY", "_TOKEN", "_PASSWORD", "_SECRET"))
        }
    )
    try:
        names = list(DEMOS) if RUN == ["all"] else RUN
        if not names or any(name not in DEMOS for name in names):
            print("Choose RUN from:", ", ".join(DEMOS), 'or ["all"]')
            return
        connection: Device
        if CLOUD:
            connection = CloudDevice(device_id=DEVICE)
        elif DEVICE.startswith(("http://", "https://")):
            connection = RemoteDevice(url=DEVICE)
        else:
            connection = AndroidDevice(serial=DEVICE)
        config = ConfigLoader.load().model_copy(deep=True)
        config.agent.workflow_timeout = TIMEOUT
        with ConfigLoader.override(config):
            async with connection as phone:
                for name in names:
                    print(f"\n=== {name} ===", flush=True)
                    try:
                        await reset_phone(phone)
                        match await DEMOS[name](phone):
                            case str() as text:
                                print(line(text))
                            case result:
                                print(line(summary(result)))
                    except Exception as error:
                        print(line(f"{type(error).__name__}: {error}"))
    except Exception as error:
        print(line(f"{type(error).__name__}: {error}"))
    finally:
        private_values.reset(token)


if __name__ == "__main__":
    asyncio.run(main())

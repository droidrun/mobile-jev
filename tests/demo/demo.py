"""Edit the block below, then run `python tests/demo/demo.py`."""

import asyncio
import json
import logging
import signal
from collections.abc import Awaitable, Callable, Collection, Iterable
from contextlib import suppress
from functools import partial
from pathlib import Path
from time import monotonic
from typing import TypeVar

import click
from mobilerun import AndroidDevice, CloudDevice, Device, RemoteDevice
from mobilerun.core.credentials import CloudCredentials, DictCredentials
from mobilerun.core.extensions import Extension
from mobilerun.core.mailbox import CloudMailbox
from mobilerun.core.operator import Operator
from mobilerun.core.phone_number import CloudPhoneNumber
from mobilerun.errors import DeviceError
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.settings import get_macro_settings
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents import (
    AgentOperator,
    AgentState,
    DecisionSelector,
    DelegatingSelector,
    LlamaIndexLLMSelector,
    LlamaIndexPlanner,
    PlanningOperator,
    tool,
)
from mobilerun_agents.agent.action_selector import Vision
from mobilerun_agents.agent.action_selector.decision_model import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.factory import build_llm
from mobilerun_agents.agent.auth import CredentialMissing
from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.agent.workflows.core.targets import target
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.settings import get_agent_settings

# --- edit me ---------------------------------------------------------
DEVICE = "4c1f61fd-5e3a-4d02-9457-478f05875ab9"  # serial, remote URL, or cloud ID
CLOUD = True
RUN = ["all"]  # choose a name in DEMOS or ["all"]
RESET = True  # stop the foreground app and press Home before each example
DEBUG = False  # True gives the CLI's per-step --debug detail
MAX_STEPS = 60
TIMEOUT = 300  # seconds per operator run; None = no deadline
MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}
COMPARE = ["openai", "anthropic"]
DEFAULT = "openai"
PLANNER = ("openai", "gpt-6-luna")
EXECUTOR = ("openai", "gpt-6-luna")
VISION_MODEL = ("openai", "gpt-6-luna")
JEV_VISION: Vision = "off"
JEV_MIN_CONFIDENCE = 0.7

GOAL = "Check the cheapest prices of Amazon and eBay prices for the game split fiction."
EASY_GOAL = "Open Settings and tell me the battery level."
VISION_GOAL = "Open Settings and describe two things visible on the screen."
JEV_GOAL = "Open Settings and report the first visible menu item without entering text."
DELEGATION_GOAL = "Open Chrome, type 'Mobilerun demo' in the search bar without submitting it."
STRUCTURED_GOAL = "Open Settings and report the battery percentage and whether it is charging."
CUSTOM_TOOL_GOAL = "Open Settings, read the battery level, and save that observation with note."
COMPARE_GOAL = EASY_GOAL

CREDENTIAL_SOURCE = "local"  # "local" (fake values below) or "cloud" (existing vault entry)
CREDENTIAL_NAME = "demo-login"
CREDENTIAL_PACKAGE = "com.android.chrome"  # vault lookup key for Chrome
CREDENTIAL_PASSWORD = "test-password-123"  # fake, local only
CREDENTIAL_GOAL = (
    "Open Chrome, tap the search bar, then call vault.type_field to paste the password "
    "of the saved credential {name} into it. Don't press enter. Report whether the field is filled."
)
PHONE_NUMBER_ID: str | None = None  # existing cloud number ID, never a new purchase
PHONE_NUMBER_TEST_CODE: str | None = None  # optional synthetic SMS code to mask in demo output
PHONE_NUMBER_GOAL = (
    "Read the attached phone number and use sms.messages to check the latest inbound SMS "
    "for a verification code. Report whether you found a code, not the code itself."
)
MAILBOX_ID: str | None = None  # existing cloud mailbox ID, never a new purchase
MAILBOX_GOAL = (
    "Use email__wait_for_code with a short timeout to check for a verification code. "
    "Report whether you found a code, not the code itself."
)
# ---------------------------------------------------------------------


class Battery(BaseModel):
    level: int = Field(ge=0, le=100)
    charging: bool


class CodeCheck(BaseModel):
    found: bool


T = TypeVar("T", bound=BaseModel)
Result = ExecutionResult[BaseModel]
Demo = Callable[[Device], Awaitable[str | Result]]
Run = Callable[[Device], Awaitable[Result]]
AGENT_CONSOLE = logging.getLogger("mobilerun_agents.agent.logs.console")


async def llm_selector(
    provider: str = DEFAULT, model: str | None = None, vision: Vision = "off"
) -> LlamaIndexLLMSelector:
    return LlamaIndexLLMSelector(
        await build_llm(provider, model or MODELS[provider]), vision=vision
    )


def decision_selector(exclude: Collection[str] = ()) -> DecisionSelector:
    return DecisionSelector(
        TypeSafeDecisionModel(),
        vision=JEV_VISION,
        exclude=exclude,
        min_confidence=JEV_MIN_CONFIDENCE,
    )


async def delegated_selector() -> DelegatingSelector:
    return DelegatingSelector(decision_selector(), await llm_selector(*VISION_MODEL, vision="on"))


async def execute(
    operator: Operator, phone: Device, goal: str, output: type[T] | None = None
) -> ExecutionResult[T]:
    return await operator.execute(phone, goal, output=output, max_steps=MAX_STEPS, timeout=TIMEOUT)


def summary(result: ExecutionResult[T]) -> str:
    return f"success={result.success} steps={result.steps} answer={result.answer or '-'}"


async def reset_phone(phone: Device) -> None:
    if not RESET:
        return
    if "stop_app" in phone.supported_operations:
        with suppress(DeviceError):
            ui = await phone.state.refresh()
            match ui.phone_state.get("packageName"):
                case str() as package if package:
                    await phone.stop_app(package)
    await phone.press_button("home")


# Run a plain LLM with the UI tree and no screenshot.
async def llm(phone: Device, goal: str = EASY_GOAL, provider: str = DEFAULT) -> Result:
    return await execute(AgentOperator(await llm_selector(provider, vision="off")), phone, goal)


# Run tree + screenshot, then screenshot only, on the same goal.
async def llm_vision(phone: Device) -> str:
    modes: tuple[Vision, ...] = ("on", "only")
    results = []
    for mode in modes:
        if mode == "only":
            await reset_phone(phone)
        operator = AgentOperator(await llm_selector(*VISION_MODEL, vision=mode))
        results.append(f"{mode}: {summary(await execute(operator, phone, VISION_GOAL))}")
    return "\n".join(results)


# Low-confidence Jev choices end the run without retrying or calling an LLM.
async def jev(phone: Device, goal: str = JEV_GOAL) -> Result:
    return await execute(AgentOperator(decision_selector()), phone, goal)


# Hide the press_button tool from Jev.
async def jev_exclude(phone: Device) -> Result:
    return await execute(
        AgentOperator(decision_selector(exclude={"press_button"})), phone, JEV_GOAL
    )


# Call the LLM only when Jev explicitly chooses delegate.
async def delegation(phone: Device, goal: str = DELEGATION_GOAL) -> Result:
    return await execute(AgentOperator(await delegated_selector()), phone, goal)


# Give the LLM planner a separate LLM executor.
async def planner_llm(phone: Device) -> Result:
    planner = LlamaIndexPlanner(await build_llm(*PLANNER))
    return await execute(PlanningOperator(planner, await llm_selector(*EXECUTOR)), phone, EASY_GOAL)


# A failed Jev subgoal goes back to the planner rather than to a delegate.
async def planner_jev(phone: Device) -> Result:
    planner = LlamaIndexPlanner(await build_llm(*PLANNER))
    return await execute(PlanningOperator(planner, decision_selector()), phone, EASY_GOAL)


# Give the planner a Jev-first executor for the hard goal.
async def planner_delegation(phone: Device, goal: str = GOAL) -> Result:
    planner = LlamaIndexPlanner(await build_llm(*PLANNER))
    return await execute(PlanningOperator(planner, await delegated_selector()), phone, goal)


async def with_extension(
    phone: Device,
    extension: Extension,
    goal: str,
    exclude: tuple[str, ...] = (),
    output: type[T] | None = None,
) -> ExecutionResult[T]:
    try:
        await extension.bind(phone)
        config = ConfigLoader.load().model_copy(deep=True)
        config.tools.exclude = [*(config.tools.exclude or []), *exclude]
        with ConfigLoader.override(config):
            return await execute(AgentOperator(await llm_selector()), phone, goal, output)
    finally:
        await extension.close()  # Base also unbinds; only newly owned resources are released.


# Use a local fake vault or an existing cloud entry without exposing its value.
async def vault_login(phone: Device) -> str:
    if CREDENTIAL_SOURCE == "local":
        vault = DictCredentials(
            {CREDENTIAL_PACKAGE: {CREDENTIAL_NAME: {"password": CREDENTIAL_PASSWORD}}}
        )
    elif CREDENTIAL_SOURCE == "cloud":
        vault = CloudCredentials()
    else:
        return 'skipped: set CREDENTIAL_SOURCE to "local" or "cloud"'
    goal = CREDENTIAL_GOAL.format(name=CREDENTIAL_NAME).replace(
        "vault.type_field", "vault__type_field"
    )
    result: Result = await with_extension(phone, vault, goal)
    return f"success={result.success} steps={result.steps}"


async def verification_status(
    phone: Device, extension: Extension, goal: str, exclude: tuple[str, ...] = ()
) -> str:
    goal += " Set output.found to true or false and leave the completion message null."
    # These examples present only a boolean; omit per-step and free-form console output.
    was_disabled = AGENT_CONSOLE.disabled
    AGENT_CONSOLE.disabled = True
    try:
        result = await with_extension(phone, extension, goal, exclude, CodeCheck)
    finally:
        AGENT_CONSOLE.disabled = was_disabled
    status = "found" if result.output and result.output.found else "not found"
    return f"{status}; steps={result.steps}"


# Temporarily attach an existing cloud number and read inbound SMS only.
async def phone_number(phone: Device) -> str:
    if not PHONE_NUMBER_ID:
        return "skipped: set PHONE_NUMBER_ID"
    if phone.cloud_id is None or "cloud_telephony" not in phone.supported_operations:
        return "skipped: use a cloud phone with telephony support"
    if "sms" in phone.extensions:
        return "skipped: use a test phone without a bound number"
    number = await CloudPhoneNumber(PHONE_NUMBER_ID)
    if number.device_id is not None:
        await number.close()
        return "skipped: use an unbound cloud number"
    goal = PHONE_NUMBER_GOAL.replace("sms.messages", "sms__messages")
    return await verification_status(phone, number, goal, exclude=("sms.send",))


# Bind an existing mailbox and wait briefly for a synthetic email code.
async def mailbox(phone: Device) -> str:
    if not MAILBOX_ID:
        return "skipped: set MAILBOX_ID"
    if "email" in phone.extensions:
        return "skipped: use a device without a bound mailbox"
    box = await CloudMailbox(MAILBOX_ID)
    return await verification_status(phone, box, MAILBOX_GOAL)


# Ask the operator to validate the result against a Pydantic model.
async def structured(phone: Device) -> str:
    result = await execute(AgentOperator(await llm_selector()), phone, STRUCTURED_GOAL, Battery)
    return result.output.model_dump_json() if result.output else summary(result)


@tool
async def note(ctx: Context[AgentState], text: str) -> str:
    """Remember an observation for the rest of this run."""
    async with ctx.store.edit_state() as state:
        state.notes.append(text)
    return "Note saved."


# Register a custom tool with a workflow-injected Context.
async def custom_tool(phone: Device) -> Result:
    return await execute(AgentOperator(await llm_selector(), tools=[note]), phone, CUSTOM_TOOL_GOAL)


def latest_checkpoint() -> Path | None:
    return max(
        get_agent_settings().runs_dir.glob("*/context.json"),
        key=lambda path: (path.stat().st_mtime_ns, path),
        default=None,
    )


# Replay the macro exported by the operator for the newest saved run.
async def macro(phone: Device) -> str:
    checkpoint = latest_checkpoint()
    if checkpoint is None:
        return "skipped: no saved run yet"
    path = (
        Path(get_macro_settings().macro_directory).expanduser()
        / checkpoint.parent.name
        / "macro.json"
    )
    if not path.is_file():
        return "skipped: the latest run did not export a macro"
    player = MacroPlayer(phone)
    recording = player.load_macro_from_file(path)
    if not recording["actions"]:
        return "skipped: no replayable actions in the latest run"
    async with asyncio.timeout(TIMEOUT):
        success = await player.replay_macro(recording, max_steps=MAX_STEPS)
    return f"{'replayed' if success else 'stopped after'} {player.executed_steps} action(s)"


# Restore the native Context and label each recorded action's target.
async def inspect_run(_: Device) -> str:
    checkpoint = latest_checkpoint()
    if checkpoint is None:
        return "skipped: no saved run yet"

    class Reader(Workflow):
        @step
        async def finish(self, ev: StartEvent) -> StopEvent:
            return StopEvent()

    ctx = Context[AgentState | PlanningState].from_dict(
        Reader(), json.loads(checkpoint.read_bytes())
    )
    state = await ctx.store.get_state()
    lines = []
    for record in state.actions:
        label = target(record, limit=None) or "-"
        if len(label) > 60:
            label = label[:59] + "…"
        lines.append(
            f"{record.step}: {record.call.tool_name} {label} | "
            f"ok={record.ok} changed={record.screen_changed}"
        )
    return "\n".join(lines) or "No actions recorded."


async def compare(phone: Device, runs: Iterable[tuple[str, Run]]) -> int:
    rows = ["setup/model\tsuccess\tsteps\tseconds\tanswer"]
    for label, run in runs:
        start = monotonic()
        success: bool | str = "-"
        steps: int | str = "-"
        try:
            await reset_phone(phone)
            result = await run(phone)
            success, steps, answer = (
                result.success,
                result.steps or 0,
                result.answer or "-",
            )
        except CredentialMissing:
            answer = "skipped (no key)"
        except Exception as error:
            success, answer = False, type(error).__name__
        answer = " ".join(answer.split())
        rows.append(f"{label}\t{success}\t{steps}\t{monotonic() - start:.1f}\t{answer}")
    print("\n".join(rows))
    return len(rows) - 1


# Compare the same goal across providers from COMPARE.
async def compare_models(phone: Device) -> str:
    runs = (
        (
            f"{provider}/{MODELS.get(provider, '?')}",
            partial(llm, goal=COMPARE_GOAL, provider=provider),
        )
        for provider in COMPARE
    )
    return f"Compared {await compare(phone, runs)} model(s)."


# Compare LLM, Jev, delegation and planner + delegation on the same goal.
async def compare_setups(phone: Device) -> str:
    runs = (
        (fn.__name__, partial(fn, goal=COMPARE_GOAL))
        for fn in (llm, jev, delegation, planner_delegation)
    )
    return f"Compared {await compare(phone, runs)} setup(s)."


DEMOS: dict[str, Demo] = {
    fn.__name__: fn
    for fn in (
        llm,
        llm_vision,
        jev,
        jev_exclude,
        delegation,
        planner_llm,
        planner_jev,
        planner_delegation,
        vault_login,
        phone_number,
        mailbox,
        structured,
        custom_tool,
        macro,
        inspect_run,
        compare_models,
        compare_setups,
    )
}


class UnknownRun(ValueError):
    """The edit block names an unavailable demo."""


async def _run_demos() -> None:
    names = list(DEMOS) if RUN == ["all"] else RUN
    unknown = [name for name in names if name not in DEMOS]
    if not names or unknown:
        raise UnknownRun(
            f"Unknown RUN selection: {unknown or names}. Choose from {', '.join(DEMOS)} or ['all']."
        )

    config = ConfigLoader.load().model_copy(deep=True)
    config.agent.workflow_timeout = TIMEOUT

    skipped = {"phone_number": PHONE_NUMBER_ID, "mailbox": MAILBOX_ID}
    active = [name for name in names if name not in skipped or skipped[name]]
    if not active:
        for name in names:
            print(f"\n=== {name} ===\nskipped: set {name.upper()}_ID")
        return

    connection: Device
    if CLOUD:
        connection = CloudDevice(device_id=DEVICE)
    elif DEVICE.startswith(("http://", "https://")):
        connection = RemoteDevice(url=DEVICE)
    else:
        connection = AndroidDevice(serial=DEVICE)
    with ConfigLoader.override(config):
        async with connection as phone:
            for name in names:
                print(f"\n=== {name} ===", flush=True)
                if name not in active:
                    print(f"skipped: set {name.upper()}_ID")
                    continue
                try:
                    if name not in {"compare_models", "compare_setups", "inspect_run"}:
                        await reset_phone(phone)
                    match await DEMOS[name](phone):
                        case str() as text:
                            print(text)
                        case result:
                            print(summary(result))
                except Exception as error:
                    print(f"failed: {type(error).__name__}")


async def main() -> None:
    with click.Context(click.Command("demo")):
        configure_cli_logging(debug=DEBUG)
        await _run_demos()


def run_main() -> None:
    console = AGENT_CONSOLE
    reported = False

    def observe_result(record: logging.LogRecord) -> bool:
        nonlocal reported
        if record.__dict__.get("agent_result") and record.getMessage().startswith("✗ cancelled"):
            reported = True
        return True

    async def run_interruptible() -> bool:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        deadline: asyncio.TimerHandle | None = None

        def cancel() -> None:
            nonlocal deadline
            task.cancel()
            if deadline is None:
                # Stop the workflow first, then bound a slow device close.
                deadline = loop.call_later(1.0, task.cancel)

        previous = signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(cancel))
        try:
            await main()
        except asyncio.CancelledError:
            return True
        finally:
            signal.signal(signal.SIGINT, previous)
            if deadline is not None:
                deadline.cancel()
        return False

    console.addFilter(observe_result)
    try:
        interrupted = asyncio.run(run_interruptible())
    except KeyboardInterrupt:
        interrupted = True
    except UnknownRun as error:
        raise SystemExit(str(error)) from None
    except Exception as error:
        raise SystemExit(type(error).__name__) from None
    finally:
        console.removeFilter(observe_result)
    if interrupted:
        if not reported:
            print("✗ cancelled")
        raise SystemExit(130) from None


if __name__ == "__main__":
    run_main()

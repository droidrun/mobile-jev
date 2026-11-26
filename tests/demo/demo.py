"""Edit the block below, then run `python tests/demo/demo.py`."""

import asyncio
import json
import os
from asyncio import sleep
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import monotonic
from typing import TypeVar

import click
from llama_index.core.llms import ChatMessage
from mobilerun import AndroidDevice, CloudDevice, Device, RemoteDevice
from mobilerun.core.credentials import CloudCredentials, DictCredentials
from mobilerun.core.extensions import Extension
from mobilerun.core.operator import Operator
from mobilerun.core.phone_number import CloudPhoneNumber
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
DEVICE = "4c1f61fd-5e3a-4d02-9457-478f05875ab9"  # serial, remote URL, or cloud ID
CLOUD = True
RUN = ["tiktok"]  # names in DEMOS below, or ["all"]
SETUPS = ["jev_only", "no_vision", "vision"]  # names in SETUP_DEMOS, or ["all"]
RESET = True  # press Home before each demo and comparison run
DEBUG = False  # True gives the CLI's per-step --debug detail
GOAL = (
    "Check the prices of Amazon and eBay for the iPhone 15 Pro Max, and tell me which is cheaper."
)
MAX_STEPS = 35
TIMEOUT = 300  # seconds per operator run; None = no deadline
MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}
COMPARE = ["openai", "anthropic"]
DEFAULT = "openai"
VISION: Vision = "off"  # default for LLM demos without an explicit mode
VISION_MODEL = ("openai", "gpt-6-luna")  # provider/model for explicit LLM vision modes
PLANNER = ("openai", "gpt-6-luna")
EXECUTOR = ("openai", "gpt-6-luna")
JEV_MIN_CONFIDENCE = 0.7
JEV_VISION: Vision = (
    "only"  # Jev in jev_delegate / jev_executor: "off" tree, "on" tree+image, "only" image + located taps
)
DELEGATE_VISION: Vision = "on"  # the LLM Jev hands steps to
MACRO: str | Path | None = None  # newest recorded macro, or specify a folder/file
CREDENTIAL_SOURCE = "local"  # "local" (fake values below) or "cloud" (existing vault entry)
CREDENTIAL_NAME = "demo-login"
CREDENTIAL_PACKAGE = "com.android.chrome"  # vault lookup key for Chrome
CREDENTIAL_PASSWORD = "test-password-123"  # fake, local only
CREDENTIAL_GOAL = (
    "Open Chrome, tap the search bar, then call vault.type_field to paste the password "
    "of the saved credential {name} into it. Don't press enter. Report whether the field is filled."
)
PHONE_NUMBER_ID: str | None = None  # existing cloud number ID, never a new purchase
PHONE_NUMBER_TEST_CODE: str | None = None  # optional synthetic SMS code to scan for
PHONE_NUMBER_GOAL = (
    "Read the attached phone number and use sms.messages to check the latest inbound SMS "
    "for a verification code. Report whether you found a code, not the code itself."
)
# Easy executor and separate TikTok device demos.
EASY_GOAL = "Open Settings and tell me the battery level."
EASY_JEV_VISION: Vision = "on"
EASY_JEV_MIN_CONFIDENCE = 0.7
EASY_MAX_STEPS = 20
TIKTOK_DEVICE: str | None = (
    "6904b14f-833d-4cdc-b0f3-320235aa25c0"  # second device: serial, remote URL or cloud ID
)
TIKTOK_CLOUD = True
TIKTOK_GOAL = (
    "Open TikTok and report the creator and caption of the first video on the For You feed."
)
TIKTOK_MAX_STEPS = 15
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
    operator: Operator,
    phone: Device,
    goal: str,
    output: type[BaseModel] | None = None,
    *,
    max_steps: int = MAX_STEPS,
) -> ExecutionResult[BaseModel]:
    # Keep the demo's redaction context outside the normal (non-debug) workflow.
    token = private_values.set(None) if not DEBUG else None
    try:
        return await operator.execute(
            phone, goal, output=output, max_steps=max_steps, timeout=TIMEOUT
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
        jev(JEV_VISION),
        await selector(*VISION_MODEL, vision=DELEGATE_VISION),
        min_confidence=JEV_MIN_CONFIDENCE,
    )


async def jev_delegate(phone: Device) -> ExecutionResult[BaseModel]:
    return await execute(AgentOperator(selector=await delegated_selector()), phone, GOAL)


async def jev_executor(phone: Device) -> ExecutionResult[BaseModel]:
    operator = PlanningOperator(
        planner=LlamaIndexPlanner(await build_llm(*PLANNER)), executor=await delegated_selector()
    )
    return await execute(operator, phone, GOAL)


async def easy_jev_executor(phone: Device) -> ExecutionResult[BaseModel]:
    planner = LlamaIndexPlanner(await build_llm(*PLANNER))
    executor = DelegatingSelector(
        jev(EASY_JEV_VISION),
        LlamaIndexLLMSelector(await build_llm(*EXECUTOR), vision="on"),
        min_confidence=EASY_JEV_MIN_CONFIDENCE,
    )
    operator = PlanningOperator(planner=planner, executor=executor)
    return await execute(operator, phone, EASY_GOAL, max_steps=EASY_MAX_STEPS)


async def easy_llm_executor(phone: Device) -> ExecutionResult[BaseModel]:
    planner = LlamaIndexPlanner(await build_llm(*PLANNER))
    operator = PlanningOperator(
        planner=planner,
        executor=LlamaIndexLLMSelector(await build_llm(*EXECUTOR), vision="on"),
    )
    return await execute(operator, phone, EASY_GOAL, max_steps=EASY_MAX_STEPS)


async def tiktok(_: Device) -> str:
    if TIKTOK_DEVICE is None:
        return "skipped: set TIKTOK_DEVICE"
    try:
        connection: CloudDevice | RemoteDevice | AndroidDevice
        if TIKTOK_CLOUD:
            connection = CloudDevice(device_id=TIKTOK_DEVICE)
        elif TIKTOK_DEVICE.startswith(("http://", "https://")):
            connection = RemoteDevice(url=TIKTOK_DEVICE)
        else:
            connection = AndroidDevice(serial=TIKTOK_DEVICE)
        async with connection as phone:
            agent = AgentOperator(selector=await selector(DEFAULT))
            result = await execute(agent, phone, TIKTOK_GOAL, max_steps=TIKTOK_MAX_STEPS)
            return summary(result)
    except Exception as error:
        return f"failed: {type(error).__name__}"


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


def report_extension_use(
    trajectory_path: str | None, label: str, expected: set[str], secrets: tuple[str, ...]
) -> None:
    names: dict[str, None] = {}
    visibility = "not checked"
    if trajectory_path:
        folder = Path(trajectory_path)
        checkpoints = (folder / "checkpoints").glob("*.json")
        last = max(
            (path for path in checkpoints if path.stem.isdecimal()),
            key=lambda path: int(path.stem),
            default=None,
        )
        macro_file = (
            Path(get_macro_settings().macro_directory).expanduser() / folder.name / "macro.json"
        )
        if last:
            for action in json.loads(last.read_bytes())["actions"]:
                names[action["name"]] = None
        if "extension_call" in names and macro_file.is_file():
            # Checkpoints can keep only the primitive name. The exported macro
            # includes the qualified name for successful, replayable calls.
            extensions = [
                action["name"]
                for action in json.loads(macro_file.read_bytes())["actions"]
                if action["action_type"] == "extension_call"
            ]
            if extensions:
                del names["extension_call"]
                names.update(dict.fromkeys(extensions))
        needles = tuple(value.encode() for value in secrets if value)
        if needles and folder.is_dir():
            files = [path for path in folder.rglob("*") if path.is_file()]
            if macro_file.is_file():
                files.append(macro_file)
            leaked = False
            for path in files:
                content = path.read_bytes()
                leaked |= any(value in content for value in needles)
            visibility = (
                "yes" if leaked else "no" if last and macro_file.is_file() else "not checked"
            )
    print(f"used: {redact(', '.join(names), set(secrets) - {''}) or '(none)'}")
    print(f"{label} tool called: {'yes' if expected.intersection(names) else 'no'}")
    print(f"secret visible in run folder: {visibility}")


async def _extension_demo(
    phone: Device,
    extension: Extension,
    goal: str,
    label: str,
    expected: set[str],
    secrets: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> str:
    known = {value for value in secrets if value}
    token = private_values.set((private_values.get() or set()) | known) if DEBUG else None
    try:
        try:
            await extension.bind(phone)
            config = ConfigLoader.load().model_copy(deep=True)
            config.checkpoints.enabled = True
            config.tools.exclude = [*(config.tools.exclude or []), *exclude]
            with ConfigLoader.override(config):
                agent = AgentOperator(selector=await selector(DEFAULT))
                result = await execute(agent, phone, goal)
        finally:
            try:
                await extension.unbind()
            finally:
                await extension.close()
        report_extension_use(result.trajectory_path, label, expected, secrets)
        return f"success={result.success} steps={result.steps}"
    except Exception as error:
        # Provider and device exceptions can carry values the demo cannot mask.
        return f"failed: {type(error).__name__}"
    finally:
        if token is not None:
            private_values.reset(token)


async def credentials(phone: Device) -> str:
    goal = CREDENTIAL_GOAL.format(name=CREDENTIAL_NAME)
    extension: Extension
    secrets: tuple[str, ...]
    if CREDENTIAL_SOURCE == "local":
        extension = DictCredentials(
            {
                CREDENTIAL_PACKAGE: {
                    CREDENTIAL_NAME: {
                        "password": CREDENTIAL_PASSWORD,
                    }
                }
            }
        )
        secrets = (CREDENTIAL_PASSWORD,)
    elif CREDENTIAL_SOURCE == "cloud":
        try:
            extension = CloudCredentials()
        except Exception as error:
            return f"failed: {type(error).__name__}"
        secrets = ()  # The public vault API never returns values to the caller.
    else:
        return 'skipped: set CREDENTIAL_SOURCE to "local" or "cloud"'
    return await _extension_demo(
        phone, extension, goal, "credential", {"vault.type_field"}, secrets
    )


async def phone_number(phone: Device) -> str:
    if not PHONE_NUMBER_ID:
        return "skipped: set PHONE_NUMBER_ID"
    if phone.cloud_id is None or "cloud_telephony" not in phone.supported_operations:
        return "skipped: use a cloud phone with telephony support"
    if "sms" in phone.extensions:
        return "skipped: use a test phone without a bound number"
    try:
        extension = await CloudPhoneNumber(PHONE_NUMBER_ID)
        if extension.device_id is not None:
            await extension.close()
            return "skipped: use an unbound cloud number"
    except Exception as error:
        return f"failed: {type(error).__name__}"
    return await _extension_demo(
        phone,
        extension,
        PHONE_NUMBER_GOAL,
        "phone number",
        {"sms.messages", "sms.wait_for_code"},
        (PHONE_NUMBER_TEST_CODE,) if PHONE_NUMBER_TEST_CODE else (),
        exclude=("sms.send",),
    )


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
        easy_jev_executor,
        easy_llm_executor,
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
        for fn in (
            compare_models,
            compare_setups,
            structured,
            custom_tool,
            macro,
            device,
            credentials,
            phone_number,
            tiktok,
        )
    },
}


async def main() -> None:
    with click.Context(click.Command("demo")):
        configure_cli_logging(debug=DEBUG)
        await _run_demos()


def run_main() -> None:
    import logging
    import signal

    console = logging.getLogger("mobilerun_agents.agent.logs.console")
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
                # The first cancellation stops the workflow; bound the subsequent device close.
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
    finally:
        console.removeFilter(observe_result)
    if interrupted:
        if not reported:
            print("✗ cancelled")
        raise SystemExit(130) from None


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
        if "phone_number" in names and not PHONE_NUMBER_ID:
            print("\n=== phone_number ===\nskipped: set PHONE_NUMBER_ID")
            names = [name for name in names if name != "phone_number"]
            if not names:
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
    run_main()

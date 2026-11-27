"""Guarded replay with independently verified, versioned step repair."""

import asyncio
import logging
from contextlib import AsyncExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar, Unpack

from mobilerun.core.devices import Device, RecordingDevice
from mobilerun.core.extensions import ActionInfo, dispatch_guard
from mobilerun.core.operator import Operator
from mobilerun.core.runs import recording_policy, recording_scope
from mobilerun.errors.operators import ExtensionActionError
from mobilerun.macro import MacroStep, build_macro, load_macro, resolve_macro_path
from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.macro.operator import MacroOperator, PlayerOptions
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.utils import compare_states, has_masked_arguments
from mobilerun.models.recording import RunStatus, RunStep
from mobilerun.models.results import ExecutionResult
from mobilerun.utils.prompts import PromptLoader
from pydantic import BaseModel, ValidationError

from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.settings import get_agent_settings
from mobilerun_agents.config.user_config import MobileConfig

from .paths import macro_identity

_PROMPTS = Path(__file__).resolve().parents[1] / "assets/prompts/agent_macro"
_LOG = logging.getLogger("mobilerun-macro")
T = TypeVar("T", bound=BaseModel)


def _commerce_action(device: Device, name: str) -> bool:
    return name.endswith(".send") or any(
        action.name == name and action.supports_billing_retry for action in device.actions.list()
    )


def _commerce(device: Device, step: MacroStep | RunStep) -> bool:
    if step.action != "extension_call":
        return False
    match step.args.get("name"):
        case str(name):
            return _commerce_action(device, name)
        case _:
            return False


def _deny_handoff_commerce(action: ActionInfo) -> None:
    if action.action_name.rsplit(".", 1)[-1] == "send" or action.supports_billing_retry:
        raise ExtensionActionError("Commerce is disabled during macro agent handoff")


def _repair_rows(run_dir: Path, step_ids: list[int]) -> list[RunStep]:
    collected = set(step_ids)
    with (run_dir / "steps.jsonl").open(encoding="utf-8") as file:
        return [row for line in file if (row := RunStep.model_validate_json(line)).i in collected]


def _publish(target: Path, run_dir: Path, steps: list[MacroStep | RunStep]) -> Path:
    # Keep unmodified source rows (including their guards, annotations, and vault references).
    # The audit trail is separate and still contains failed attempts.
    name, version = macro_identity(target)
    now = datetime.now(timezone.utc)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir) as splice:
        for i, step in enumerate(steps, 1):
            row = RunStep(
                i=i,
                started_at=now,
                finished_at=now,
                action=step.action,
                args=step.args,
                ok=True,
                before=step.before,
                after=step.after,
                note=step.note,
                target=step.target,
                by="macro",
            )
            splice.write(row.model_dump_json() + "\n")
        splice.flush()
        return build_macro(Path(splice.name), name, after_version=version)


class MacroAgentOperator(MacroOperator):
    """Replay a versioned macro on a caller-owned device, repairing one step at a time."""

    def __init__(
        self,
        macro_file: str | Path | None = None,
        agent: Operator | None = None,
        config: MobileConfig | None = None,
        start_from_step: int = 0,
        record: bool | None = None,
        **player_options: Unpack[PlayerOptions],
    ) -> None:
        super().__init__(macro_file, start_from_step=start_from_step, **player_options)
        self._agent = agent
        self._config = config
        self._record = record

    async def _handoff(
        self,
        device: Device,
        prompt: str,
        deadline: asyncio.Timeout,
        remaining: int | None,
        record: bool,
        output: type[T] | None = None,
    ) -> ExecutionResult[T]:
        if device._extension_state.sensitive_ui:
            return ExecutionResult(
                success=False,
                answer="Device may contain sensitive input",
                device=type(device).__name__,
            )
        agent = self._agent
        commerce = {
            entry.name
            for entry in device.actions.list(audience="operator")
            if entry.name.endswith(".send") or entry.supports_billing_retry
        }
        config = self._config if agent is None else None
        if commerce:
            config = config or ConfigLoader.load()
            tools = config.tools.model_copy(
                update={"exclude": sorted(set(config.tools.exclude or ()) | commerce)}
            )
            config = config.model_copy(update={"tools": tools})
        if agent is None:
            from mobilerun_agents.agent.operators import AgentOperator

            agent = AgentOperator(record=record)
        end = deadline.when()
        with (
            ConfigLoader.override(config) if config is not None else nullcontext()
        ), recording_scope(by="agent"), dispatch_guard(_deny_handoff_commerce):
            return await agent.execute(
                device,
                prompt,
                output=output,
                max_steps=remaining,
                timeout=None if end is None else max(0.0, end - asyncio.get_running_loop().time()),
            )

    async def execute(
        self,
        device: Device,
        instruction: str,
        *,
        output: type[T] | None = None,
        max_steps: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult[T]:
        if self.start_from_step < 0 or (max_steps is not None and max_steps < 0):
            raise MacroError("Step limits must not be negative")
        target = resolve_macro_path(self.macro_file or instruction)
        steps = load_macro(target)
        end = None if max_steps is None else self.start_from_step + max_steps
        selected = steps[self.start_from_step : end]
        device_name = type(device).__name__
        enabled = self._record
        if enabled is None:
            enabled = get_agent_settings().MOBILERUN_AGENT_RECORDING_ENABLED
        if enabled is None:
            enabled = (self._config or ConfigLoader.load()).recording.enabled
        recording = enabled and device.recording_allowed() and recording_policy()[0]
        recorder = device.recording_device() if recording else None
        owned = recording and recorder is None
        if owned:
            recorder = RecordingDevice(
                device,
                kind="macro_replay",
                source_macro=str(target),
                by="macro",
                capture="full" if any(step.after is not None for step in selected) else "actions",
            )
            device = recorder
        status: RunStatus = "failed"
        active_step: int | None = None
        run_dir: Path | None = None
        used = 0
        commerce_dispatched = False
        repaired = False
        publishable = selected == steps
        splice: list[RunStep | MacroStep] = []

        def result(success: bool, answer: str, value: T | None = None) -> ExecutionResult[T]:
            return ExecutionResult(
                success=success,
                answer=answer,
                output=value,
                steps=used,
                device=device_name,
                trajectory_path=str(run_dir) if run_dir is not None else None,
            )

        player = MacroPlayer(device, **self.player_options)
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline, AsyncExitStack() as replay:
                replay.enter_context(recording_scope(enabled=recording))
                replay.enter_context(player.replay_scope())
                if recorder is not None and selected and device.recording_allowed():
                    run_dir = recorder.start_run()
                # Strict replay uses this same validation for every selected row
                # before dispatch. play_step also revalidates against current bindings.
                for step in selected:
                    active_step = step.i
                    player._preflight(step)
                for position, step in enumerate(selected):
                    active_step = step.i
                    remaining = None if max_steps is None else max_steps - used
                    if remaining is not None and remaining <= 0:
                        return result(False, "Macro replay ran out of steps")
                    if position:
                        await asyncio.sleep(player.delay_between_actions)
                    attempt_ids: list[int] = []
                    capture = (
                        recorder.collect_steps()
                        if recorder is not None
                        else nullcontext(attempt_ids)
                    )
                    try:
                        with capture as attempt_ids, recording_scope(by="macro"):
                            await player.play_step(step)
                    except MacroMismatch as mismatch:
                        dispatched = mismatch.phase != "before"
                        if dispatched:
                            used += 1
                            commerce_dispatched |= _commerce(device, step)
                        reason = None
                        if player.tainted:
                            reason = "credential replay may have left sensitive input visible"
                        elif device._extension_state.sensitive_ui:
                            reason = "device may contain sensitive input"
                        elif step.after is None:
                            reason = 'no after fingerprint; record the macro with capture="full"'
                        elif commerce_dispatched or _commerce(device, step):
                            reason = "commerce may have been dispatched or is required by this step"
                        elif recorder is None or run_dir is None:
                            reason = "recording is disabled"
                        elif dispatched and not attempt_ids:
                            reason = "the dispatched attempt was not recorded"
                        elif max_steps is not None and used >= max_steps:
                            reason = "step budget exhausted"
                        if reason is not None:
                            return result(False, f"{mismatch}; repair refused: {reason}")
                        assert (
                            step.after is not None and recorder is not None and run_dir is not None
                        )
                        entry_matches = not dispatched or (
                            step.before is not None
                            and compare_states(
                                step.before,
                                await recorder.capture_fingerprint(fresh=True),
                                player.state_threshold,
                            )
                        )
                        prompt = await PromptLoader.load_prompt(
                            str(_PROMPTS / "diverged.jinja2"),
                            {"step": step.i, "target": step.after.model_dump_json()},
                        )
                        with recorder.collect_steps() as repair_ids:
                            repair: ExecutionResult[BaseModel] = await self._handoff(
                                device,
                                prompt,
                                deadline,
                                None if max_steps is None else max_steps - used,
                                recording,
                            )
                        used += repair.steps if repair.steps is not None else len(repair_ids)
                        if not repair.success or (max_steps is not None and used > max_steps):
                            return result(False, f"{mismatch}; agent repair failed")
                        if not compare_states(
                            step.after,
                            await recorder.capture_fingerprint(fresh=True),
                            player.state_threshold,
                        ):
                            return result(
                                False, f"{mismatch}; repair did not reach the target fingerprint"
                            )
                        repaired = True
                        repair_rows = _repair_rows(run_dir, repair_ids)
                        commerce_dispatched |= any(_commerce(device, row) for row in repair_rows)
                        rows = [row for row in repair_rows if row.ok]
                        if (
                            commerce_dispatched
                            or not rows
                            or not entry_matches
                            or any(row.before is None or row.after is None for row in rows)
                            or any(has_masked_arguments(row.args) for row in rows)
                            or any(
                                not row.ok and (row.before is None or row.before != row.after)
                                for row in repair_rows
                            )
                            or (
                                dispatched
                                and (
                                    step.before is None
                                    or not compare_states(
                                        step.before, rows[0].before, player.state_threshold
                                    )
                                )
                            )
                        ):
                            publishable = False
                        if rows:
                            # The last action must re-establish the original, verified postcondition.
                            rows[-1] = rows[-1].model_copy(update={"after": step.after})
                            splice.extend(rows)
                    else:
                        used += 1
                        commerce_dispatched |= _commerce(device, step)
                        splice.append(step)
                active_step = None
                value = None
                answer = "Macro replay completed"
                if output is not None:
                    if player.tainted:
                        return result(False, "Output extraction refused after credential replay")
                    if device._extension_state.sensitive_ui:
                        return result(
                            False, "Output extraction refused: device may contain sensitive input"
                        )
                    remaining = None if max_steps is None else max_steps - used
                    if remaining is not None and remaining <= 0:
                        return result(
                            False, "Macro replay ran out of steps before output extraction"
                        )
                    goal = (
                        instruction
                        if self.macro_file
                        else await PromptLoader.load_prompt(str(_PROMPTS / "fallback_goal.jinja2"))
                    )
                    prompt = await PromptLoader.load_prompt(
                        str(_PROMPTS / "extract.jinja2"), {"goal": goal}
                    )
                    extraction = await self._handoff(
                        device, prompt, deadline, remaining, recording, output
                    )
                    used += extraction.steps or 0
                    if not extraction.success or (max_steps is not None and used > max_steps):
                        return result(False, "Macro output extraction failed")
                    value = extraction.output
                    answer = extraction.answer or answer
                if repaired:
                    version = None
                    if publishable and run_dir is not None:
                        try:
                            version = _publish(target, run_dir, splice)
                        except (MacroError, OSError, ValidationError):
                            pass
                    if version is None:
                        answer = "Macro recovered; a new version was not published"
                        _LOG.warning(answer)
                    else:
                        answer = f"Macro repaired and published as {version.name}"
                        _LOG.info(answer)
                status = "succeeded"
                return result(True, answer, value)
        except (asyncio.CancelledError, KeyboardInterrupt):
            status = "cancelled"
            raise
        finally:
            if owned and recorder is not None:
                recorder.finish(status, failed_step=active_step)

import asyncio
import json
from pathlib import Path

import pytest
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.credentials.masking import current_masks
from mobilerun.core.devices import Device, RecordingDevice, StealthDevice
from mobilerun.core.extensions import Extension, action
from mobilerun.core.runs import recording_scope
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors.operators import ExtensionActionError
from mobilerun.macro import MacroError, MacroOperator, build_macro, load_macro
from mobilerun.macro.errors import MacroMismatch
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun_agents.config.user_config import MobileConfig
from mobilerun_agents.macro_agent import MacroAgentOperator


class Screen(StateProvider):
    async def get_state(self):
        return UIState([], "", "", {"packageName": self._device.screen}, 10, 10)


class Phone(Device):
    def __init__(self, screen="home", broken_screen="wrong"):
        self.screen = screen
        self.broken_screen = broken_screen
        self.calls = []
        super().__init__(state_provider=Screen(self))

    @property
    def platform(self):
        return "android"

    async def screenshot(self, hide_overlay=True):
        return self.screen.encode()

    async def tap(self, x, y):
        self.calls.append(x)
        self.screen = {1: self.broken_screen, 2: "repairing", 3: "ready", 4: "done"}[x]


class Repair:
    def __init__(self, taps=(2, 3), success=True):
        self.taps = taps
        self.success = success
        self.calls = []

    async def execute(self, device, instruction, **options):
        self.calls.append((device, instruction, options))
        for x in self.taps:
            await device.tap(x, 0)
        return ExecutionResult(success=self.success, device="fake", steps=len(self.taps))


def fp(app):
    return dict(app=app, activity=None, elements=[])


def tap(i, x, before=None, after=None, **details):
    return dict(i=i, action="tap", args=dict(x=x, y=0), before=before, after=after, **details)


def repair_macro(path, phone, repair, **options):
    return asyncio.run(
        MacroAgentOperator(path, agent=repair, state_timeout=0, delay_between_actions=0).execute(
            phone, "A broader goal", **options
        )
    )


def manifest(tmp_path):
    (run,) = (tmp_path / "runs").iterdir()
    return json.loads((run / "run.json").read_text(encoding="utf-8"))


def record_macro(capture):
    async def run():
        recorder = RecordingDevice(Phone(broken_screen="ready"), capture=capture)
        await recorder.tap(1, 0)
        await recorder.tap(4, 0)
        recorder.finish("succeeded")
        return build_macro(recorder.run_dir / "steps.jsonl", "inbox")

    return asyncio.run(run())


def test_full_recording_repairs_and_publishes_next_version(macros):
    source = record_macro("full")
    assert load_macro(source)[0].after.app == "ready"
    phone = Phone(broken_screen="home")
    repair = Repair()

    result = repair_macro(source, phone, repair)

    assert result.success and phone.calls == [1, 2, 3, 4]
    assert "published as v2.jsonl" in result.answer
    assert [row.args["x"] for row in load_macro(source.with_name("v2.jsonl"))] == [2, 3, 4]
    replay_dir = Path(result.trajectory_path)
    rows = [json.loads(line) for line in (replay_dir / "steps.jsonl").read_text().splitlines()]
    agent_rows = [row for row in rows if row["by"] == "agent"]
    assert len(agent_rows) == 2
    assert all(set(row["screens"]) == {"before", "after"} for row in agent_rows)
    assert all(row["before"] and row["after"] for row in agent_rows)
    assert (replay_dir / agent_rows[-1]["screens"]["after"]["png"]).read_bytes() == b"ready"


@pytest.mark.anyio
async def test_repaired_macro_excludes_concurrent_agent_rows(write_macro):
    entered, release = asyncio.Event(), asyncio.Event()

    class ConcurrentRepair(Repair):
        async def execute(self, device, instruction, **options):
            await device.tap(2, 0)
            entered.set()
            await release.wait()
            await device.tap(3, 0)
            return ExecutionResult(success=True, device="fake")

    source = write_macro([tap(1, 1, fp("home"), fp("ready"))])
    recorder = RecordingDevice(Phone(broken_screen="home"))
    task = asyncio.create_task(
        MacroAgentOperator(source, agent=ConcurrentRepair(), state_timeout=0).execute(
            recorder, "repair"
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with recording_scope(by="agent"):
            await recorder.tap(4, 0)
    finally:
        release.set()
        result = await task
    assert result.success
    assert [step.args["x"] for step in load_macro(source.with_name("v4.jsonl"))] == [2, 3]


@pytest.mark.parametrize("opt_out", ["config", "env", "operator", "stealth"])
def test_recording_opt_out_refuses_repair_without_artifacts(
    write_macro, tmp_path, monkeypatch, opt_out
):
    phone = Phone(broken_screen="home")
    repair = Repair()
    source = write_macro([tap(1, 1, fp("home"), fp("ready"))])
    if opt_out == "env":
        monkeypatch.setenv("MOBILERUN_AGENT_RECORDING_ENABLED", "false")
    result = asyncio.run(
        MacroAgentOperator(
            source,
            agent=repair,
            state_timeout=0,
            config=MobileConfig(recording={"enabled": False}) if opt_out == "config" else None,
            **({"record": False} if opt_out == "operator" else {}),
        ).execute(StealthDevice(phone) if opt_out == "stealth" else phone, "repair")
    )
    assert not result.success and "recording is disabled" in result.answer
    assert not repair.calls and not (tmp_path / "runs").exists()
    assert not source.with_name("v4.jsonl").exists()


def test_actions_only_macro_replays_but_cannot_be_repaired(macros):
    class BrokenPhone(Phone):
        async def tap(self, x, y):
            raise OSError("dispatch failed")

    source = record_macro("actions")
    assert all(row.before is None and row.after is None for row in load_macro(source))
    phone = Phone()
    replay = MacroOperator(delay_between_actions=0)
    assert asyncio.run(replay.execute(phone, str(source))).success and phone.calls == [1, 4]
    with pytest.raises(MacroMismatch):
        asyncio.run(replay.execute(BrokenPhone(), str(source)))

    repair = Repair()
    result = repair_macro(source, BrokenPhone(), repair)

    assert not result.success and not repair.calls
    assert "repair refused" in result.answer and 'capture="full"' in result.answer
    assert list(source.parent.glob("v*.jsonl")) == [source]
    assert not (Path(result.trajectory_path) / "screens").exists()


def test_repair_respects_caller_actions_only_recorder(write_macro):
    source = write_macro([tap(7, 1, fp("original"), fp("ready"))])
    phone = Phone()
    recorder = RecordingDevice(phone, capture="actions")
    repair = Repair(taps=(3,))

    result = repair_macro(source, recorder, repair)

    assert result.success and phone.calls == [3]
    assert repair.calls[0][0] is recorder
    (line,) = (Path(result.trajectory_path) / "steps.jsonl").read_text().splitlines()
    row = json.loads(line)
    assert row["by"] == "agent" and not row.get("before") and not row.get("after")
    assert not row.get("screens")
    assert "not published" in result.answer and not source.with_name("v4.jsonl").exists()


def test_repair_verifies_the_recorders_final_state_not_the_transient_screen(write_macro):
    class TransientScreen(Screen):
        async def get_final_state(self):
            return UIState([], "", "", {"packageName": "wrong"}, 10, 10)

    phone = Phone()
    recorder = RecordingDevice(phone, state_provider=TransientScreen(phone))
    source = write_macro([tap(1, 1, fp("original"), fp("ready"))])
    result = repair_macro(source, recorder, Repair(taps=(3,)))

    assert phone.screen == "ready"
    assert not result.success and "did not reach the target" in result.answer
    assert not source.with_name("v4.jsonl").exists()


@pytest.mark.parametrize("latest", [3, 5])
def test_repair_excludes_dispatched_attempt_and_splices_before_replaying_next_row(
    write_macro, latest
):
    rows = [tap(7, 1, fp("home"), fp("ready")), tap(8, 4, fp("ready"), note="preserve me")]
    source = write_macro(rows)
    existing = source if latest == 3 else write_macro(rows, latest)
    existing_bytes = existing.read_bytes()
    source_bytes = source.read_bytes()
    phone = Phone(broken_screen="home")
    repair = Repair()

    result = repair_macro(source, phone, repair, max_steps=4, timeout=30)

    assert result.success and result.steps == 4
    assert phone.calls == [1, 2, 3, 4]
    ((device, prompt, options),) = repair.calls
    assert isinstance(device, RecordingDevice)
    assert '"ready"' in prompt and "A broader goal" not in prompt
    assert options["max_steps"] == 3 and 0 < options["timeout"] <= 30
    assert options["output"] is None
    run_rows = [
        json.loads(line)
        for line in (Path(result.trajectory_path) / "steps.jsonl").read_text().splitlines()
    ]
    assert [(row["by"], row["ok"]) for row in run_rows] == [
        ("macro", True),
        ("agent", True),
        ("agent", True),
        ("macro", True),
    ]
    new_version = source.with_name(f"v{latest + 1}.jsonl")
    repaired = load_macro(new_version)
    assert [row.args["x"] for row in repaired] == [2, 3, 4]
    assert repaired[-1].note == "preserve me"
    assert existing.read_bytes() == existing_bytes and source.read_bytes() == source_bytes
    fresh = Phone()
    strict_result = asyncio.run(
        MacroOperator(state_timeout=0, delay_between_actions=0).execute(fresh, str(new_version))
    )
    assert strict_result.success and fresh.calls == [2, 3, 4]


@pytest.mark.parametrize("agent_success", [True, False])
def test_repair_requires_agent_success_and_independently_verified_target(
    write_macro, tmp_path, agent_success
):
    source = write_macro([tap(7, 1, fp("original"), fp("ready")), tap(8, 4)])
    repair = Repair(taps=(2,) if agent_success else (3,), success=agent_success)
    phone = Phone()

    assert not repair_macro(source, phone, repair).success
    assert phone.calls == ([2] if agent_success else [3])
    record = manifest(tmp_path)
    assert record["status"] == "failed" and record["failed_step"] == 7
    assert record["source_macro"] == str(source)
    assert list(source.parent.glob("v*.jsonl")) == [source]


def test_before_guard_repair_can_publish_without_dispatching_the_mismatched_row(write_macro):
    source = write_macro([tap(1, 1, fp("original"), fp("ready")), tap(2, 4)])
    phone = Phone()
    result = repair_macro(source, phone, Repair(taps=(3,)), max_steps=2)
    assert result.success and phone.calls == [3, 4]
    assert [row.args["x"] for row in load_macro(source.with_name("v4.jsonl"))] == [3, 4]


@pytest.mark.parametrize("case", ["unsafe entry", "budget", "start offset"])
def test_recovery_without_complete_safe_replay_does_not_publish(write_macro, case):
    rows = [tap(7, 1, fp("home"), fp("ready")), tap(8, 4)]
    source = write_macro([tap(6, 4), *rows] if case == "start offset" else rows)
    phone = Phone(broken_screen="wrong" if case == "unsafe entry" else "home")
    repair = Repair(taps=(3,))
    result = asyncio.run(
        MacroAgentOperator(
            source,
            agent=repair,
            state_timeout=0,
            delay_between_actions=0,
            start_from_step=int(case == "start offset"),
        ).execute(phone, "A broader goal", max_steps=2 if case == "budget" else None)
    )
    assert phone.calls == ([1, 3] if case == "budget" else [1, 3, 4])
    assert result.success == (case != "budget")
    if case == "unsafe entry":
        assert "not published" in result.answer
    assert not source.with_name("v4.jsonl").exists()


@pytest.mark.parametrize("reason", ["missing target", "stealth", "commerce", "earlier commerce"])
def test_unsafe_repair_is_refused_without_agent_dispatch(write_macro, tmp_path, reason):
    class SMS(Extension):
        default_name = "sms"

        @action()
        async def send(self) -> bool:
            """Simulate one potentially billable send."""
            sent.append(True)
            return True

    async def run():
        await SMS().bind(phone)
        return await MacroAgentOperator(
            source, agent=repair, state_timeout=0, delay_between_actions=0
        ).execute(StealthDevice(phone) if reason == "stealth" else phone, "A broader goal")

    sent = []
    phone = Phone()
    repair = Repair()
    step = tap(7, 1, fp("original"), None if reason == "missing target" else fp("ready"))
    sms = dict(i=6, action="extension_call", args={"name": "sms.send", "arguments": {}})
    rows = [step]
    if reason == "commerce":
        rows = [dict(sms, i=7, after=fp("ready"))]
    elif reason == "earlier commerce":
        rows.insert(0, sms)
    source = write_macro(rows)
    assert not asyncio.run(run()).success
    assert not repair.calls and not phone.calls
    assert len(sent) == int("commerce" in reason)
    assert not source.with_name("v4.jsonl").exists()
    if reason == "stealth":
        assert not (tmp_path / "runs").exists()
    else:
        assert manifest(tmp_path)["failed_step"] == 7


@pytest.mark.parametrize("extraction", [False, True])
def test_preexisting_device_taint_blocks_agent_handoff(write_macro, extraction):
    class Output(BaseModel):
        title: str

    phone = Phone()
    phone._extension_state.sensitive_ui = True
    agent = Repair()
    source = write_macro([tap(7, 4)] if extraction else [tap(7, 1, fp("original"), fp("ready"))])
    result = repair_macro(source, phone, agent, output=Output if extraction else None)
    assert not result.success and "sensitive" in result.answer
    assert not agent.calls and phone.calls == ([4] if extraction else [])


def test_matching_macro_can_extract_output_without_repeating_it(write_macro):
    class Output(BaseModel):
        title: str

    class Extract:
        async def execute(self, device, instruction, **options):
            assert "without repeating" in instruction
            assert device.recording_device() is not None
            return ExecutionResult(
                success=True, device="fake", output=Output(title="Inbox"), steps=1
            )

    source = write_macro([tap(1, 4)])
    phone = Phone()
    result = repair_macro(source, phone, Extract(), output=Output)
    assert result.output.title == "Inbox" and result.steps == 2
    assert phone.calls == [4] and not source.with_name("v4.jsonl").exists()


def test_successful_repair_followed_by_terminal_failure_does_not_publish(write_macro, tmp_path):
    source = write_macro([tap(7, 1, fp("original"), fp("ready")), tap(8, 4, fp("unexpected"))])
    phone = Phone()
    result = repair_macro(source, phone, Repair(taps=(3,)))
    assert not result.success and phone.calls == [3]
    assert not source.with_name("v4.jsonl").exists()
    assert manifest(tmp_path)["failed_step"] == 8


def test_failed_action_inside_repair_cannot_be_omitted_if_it_changed_the_screen(write_macro):
    class ChangedOnError(Phone):
        async def tap(self, x, y):
            await super().tap(x, y)
            if x == 2:
                raise ValueError("changed the screen but failed")

    class Retry(Repair):
        async def execute(self, device, instruction, **options):
            try:
                await device.tap(2, 0)
            except ValueError:
                pass
            return await super().execute(device, instruction, **options)

    source = write_macro([tap(7, 1, fp("original"), fp("ready")), tap(8, 4)])
    phone = ChangedOnError()
    result = repair_macro(source, phone, Retry(taps=(3,)))
    assert result.success and phone.calls == [2, 3, 4]
    assert not source.with_name("v4.jsonl").exists()


def test_repair_timeout_persists_original_failed_index(write_macro, tmp_path):
    class NeverCompletes(Repair):
        async def execute(self, device, instruction, **options):
            async with asyncio.timeout(0):
                await asyncio.Event().wait()

    source = write_macro([tap(7, 1, fp("original"), fp("ready"))])
    with pytest.raises(TimeoutError):
        repair_macro(source, Phone(), NeverCompletes(), timeout=30)
    assert manifest(tmp_path)["status"] == "failed"
    assert manifest(tmp_path)["failed_step"] == 7
    assert not source.with_name("v4.jsonl").exists()


@pytest.mark.parametrize("extraction", [False, True])
@pytest.mark.parametrize("action_name", ["sms.send", "sms.purchase", "sms.notify"])
def test_agent_handoff_blocks_commerce_before_dispatch(write_macro, extraction, action_name):
    dispatched = []

    class SMS(Extension):
        default_name = "sms"

        async def _with_billing(self, policy) -> bool:
            dispatched.append("purchase")
            return True

        @action(billing_retry=_with_billing)
        async def purchase(self) -> bool:
            """Potentially billable operation exposed under a custom name."""
            dispatched.append("purchase")
            return True

        @action()
        async def send(self) -> bool:
            """Send an SMS."""
            dispatched.append("send")
            return True

        @action()
        async def notify(self) -> bool:
            """Send SMS through a nested action not identified by the outer name."""
            return await self.send()

    class Output(BaseModel):
        title: str

    class CommerceAttempt(Repair):
        async def execute(self, device, instruction, **options):
            with pytest.raises(ExtensionActionError, match="Commerce is disabled"):
                await device.actions.call(action_name, {})
            if extraction:
                return ExecutionResult(success=True, device="fake", output=Output(title="Inbox"))
            return await super().execute(device, instruction, **options)

    async def run():
        phone = Phone()
        await SMS().bind(phone)
        result = await MacroAgentOperator(
            source, agent=CommerceAttempt(taps=(3,)), state_timeout=0, delay_between_actions=0
        ).execute(phone, "A broader goal", output=Output if extraction else None)
        assert result.success and not dispatched
        # The guard is scoped to handoff, not caller use of the same action.
        assert await phone.actions.call(action_name, {}) is True
        assert dispatched == ["purchase" if action_name == "sms.purchase" else "send"]

    source = write_macro([tap(7, 4)] if extraction else [tap(7, 1, fp("original"), fp("ready"))])
    asyncio.run(run())


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ({"action": "input_text", "args": {"text": "***"}}, "redacted action"),
        ({"action": "type_secret", "args": {"protected": True}}, "Protected credential"),
    ],
)
def test_assisted_preflight_rejects_later_row_before_any_mutation(
    write_macro, tmp_path, invalid, error
):
    source = write_macro([tap(7, 4), {"i": 8, **invalid}])
    phone = Phone()
    repair = Repair()

    with pytest.raises(MacroError, match=error):
        repair_macro(source, phone, repair)
    assert not phone.calls and not repair.calls
    assert manifest(tmp_path)["failed_step"] == 8
    assert not source.with_name("v4.jsonl").exists()


@pytest.mark.parametrize("mismatch", [True, False])
def test_vault_write_blocks_later_repair_or_extraction(write_macro, tmp_path, mismatch):
    class VaultPhone(Phone):
        async def _input_secret(self, value):
            assert value == "fixture-value"
            self.calls.append("vault write")
            return True

        async def tap(self, x, y):
            masks = current_masks.get()
            assert masks is not None and masks.active
            await super().tap(x, y)

    class Output(BaseModel):
        title: str

    async def run():
        phone = VaultPhone()
        await DictCredentials({"login": "fixture-value"}).bind(phone)
        op = MacroAgentOperator(source, agent=repair, state_timeout=0, delay_between_actions=0)
        if mismatch:
            with pytest.raises(MacroError, match="credential replay; handoff refused"):
                await op.execute(phone, "goal")
        else:
            result = await op.execute(phone, "goal", output=Output)
            assert not result.success and "credential" in result.answer
        assert phone.calls == (["vault write"] if mismatch else ["vault write", 4])

    rows = [{"i": 7, "action": "type_secret", "args": {"name": "login"}}]
    rows.append(tap(8, 4, fp("original"), fp("ready")) if mismatch else tap(8, 4))
    source = write_macro(rows)
    repair = Repair()
    asyncio.run(run())
    assert not repair.calls and not source.with_name("v4.jsonl").exists()
    assert manifest(tmp_path)["status"] == "failed"
    if mismatch:
        assert manifest(tmp_path)["failed_step"] == 8

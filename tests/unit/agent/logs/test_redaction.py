import asyncio
from datetime import UTC, datetime

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem

from mobilerun_agents.agent.logs.redaction import Redactor, observe_vault
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from tests.unit.device_fixtures import MemoryDevice


def test_masked_spans_include_registered_variants_and_overlapping_keys():
    redact = Redactor()
    redact.add("fixture alpha")
    redact.add("fixture-token suffix")

    text = "fixture alpha / fixture%20alpha / Bearer fixture-token suffix"
    assert redact.masked_spans(text) == [(0, 13), (16, 31), (34, 54), (41, 61)]
    assert redact.masked_spans("ordinary text") == []
    assert redact.text(text) == "*** / *** / ***"


def test_overlapping_vault_scopes_mask_only_their_own_values_and_restore_device():
    first_secret, second_secret = "first-vault-fixture", "second-vault-fixture"
    first, second = Redactor(), Redactor()

    class Phone(MemoryDevice):
        async def _input_secret(self, value: str) -> bool:
            self.inputs.append(value)
            return True

    async def run():
        phone = Phone()
        original = phone._input_secret
        vault = await DictCredentials({"first": first_secret, "second": second_secret}).bind(phone)
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        first_exited = asyncio.Event()

        async def first_run():
            with observe_vault(phone, first):
                first_entered.set()
                await second_entered.wait()
                await vault.type_field(None, "password", "first")
            first_exited.set()

        async def second_run():
            await first_entered.wait()
            with observe_vault(phone, second):
                second_entered.set()
                await first_exited.wait()
                await vault.type_field(None, "password", "second")

        try:
            await asyncio.wait_for(asyncio.gather(first_run(), second_run()), timeout=5)
            assert phone.inputs == [first_secret, second_secret]
            assert first.text(first_secret) == "***"
            assert first.text(second_secret) == second_secret
            assert second.text(second_secret) == "***"
            assert second.text(first_secret) == first_secret
            assert phone._input_secret == original
        finally:
            await vault.unbind()

    asyncio.run(run())


def test_device_context_payload_is_masked_without_changing_checkpoint_structure():
    at = datetime(2026, 11, 26, 17, 42, tzinfo=UTC)
    snapshot = StepSnapshot(
        at=at,
        ui=UIState([], "", "", {}, 1742, 100),
        device_context=[
            ContextItem(
                source="fixture-extension",
                kind="fixture-kind",
                data={
                    "nested": {"pin": 1742, "message": "fixture-secret", "at": "fixture-secret"},
                    "fixture-key": "public",
                },
            )
        ],
    )
    redact = Redactor()
    redact.add([1742, "fixture-secret", "fixture-key", "source", "at"])

    payload = redact.value({"before": snapshot.model_dump(mode="json")}, freeform=False)
    restored = StepSnapshot.model_validate(payload["before"])
    assert restored.at == at
    assert restored.ui.screen_width == 1742
    [item] = restored.device_context
    assert item.source == "fixture-extension"
    assert item.kind == "fixture-kind"
    assert item.data == {"nested": {"pin": "***", "message": "***", "***": "***"}, "***": "public"}


def test_device_context_non_dict_entries_use_generic_redaction():
    redact = Redactor()
    redact.add("fixture-secret")

    payload = {"device_context": [None, ["fixture-secret"], {"data": {"at": "fixture-secret"}}]}
    assert redact.value(payload, freeform=False) == {
        "device_context": [None, ["***"], {"data": {"at": "***"}}]
    }

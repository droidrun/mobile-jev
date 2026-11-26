import json
import socket

import pytest
from llama_index.core.base.llms.types import ImageBlock
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.player import MacroPlayer

from mobilerun_agents.agent.checkpoints.store import RunRecorder
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from tests.unit.device_fixtures import MemoryDevice


def record(step, name, args, **extra):
    return ActionRecord(
        **(
            dict(
                step=step,
                name=name,
                args=args,
                ok=True,
                summary="ok",
                before=StepSnapshot(
                    goal="", step=step, ui=UIState([], "", "", {}, 0, 0), device_context=[]
                ),
            )
            | extra
        )
    )


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network is disabled")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


async def test_checkpoint_snapshots_event_step_and_exports_recorded_pre_states(tmp_path):
    folder = tmp_path / "run"
    macro_path = tmp_path / "macros/run/macro.json"
    image = b"captured PNG"
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(image)
    nodes = [{"text": "a" * 20000}]
    snapshot = StepSnapshot(
        goal="search",
        step=1,
        ui=UIState(nodes, "", "", {}, 0, 0),
        device_context=[],
    )
    tap = record(1, "tap", {"x": 3, "y": 4}, before=snapshot)
    typed = record(
        1,
        "input_text",
        {"text": 'example "text"'},
        summary='typed example "text"',
        before=snapshot.model_copy(
            update={
                "ui": UIState([{"focused": True}], "", "", {}, 0, 0),
                "screenshot": ImageBlock(path=image_path),
            }
        ),
    )
    later = record(2, "wait", {"duration": 0}, ok=False)
    state = AgentState(goal="search", step=2, actions=[tap, typed, later])
    recorder = RunRecorder(folder, macro_path=macro_path)
    assert not folder.exists()
    await recorder.on_step(tap)
    await recorder.on_step(typed)
    checkpoint = folder / "checkpoints/0001.json"
    assert checkpoint.stat().st_size < 2000
    saved = json.loads(checkpoint.read_text())
    entries = saved["actions"]
    assert saved["step"] == 1
    assert len(entries) == 2
    assert entries[1]["args"] == {}
    assert 'example "text"' not in entries[1]["summary"]
    assert entries[1]["screenshot"] == str(image_path)
    assert image_path.read_bytes() == image
    assert list(folder.rglob("*.png")) == []
    assert entries[0]["ui_state"] == entries[1]["ui_state"] == "ui_states/0001.json"
    # Inspection files aren't an input to macro export.
    (folder / entries[0]["ui_state"]).unlink()
    await recorder.finish(state)
    macro = MacroPlayer().load_macro_from_file(macro_path)
    assert macro["description"] == "search" and macro["total_actions"] == 2
    tap_action, typed_action = macro["actions"]
    assert tap_action["action_type"] == "tap" and tap_action["x"] == 3
    assert tap_action["pre_state"]["nodes"][0]["text"] == nodes[0]["text"]
    assert typed_action["pre_state"]["nodes"][0]["focused"] is True
    assert typed_action["text"] == typed.args["text"]


async def test_player_types_plain_text_and_vault_action_exports_only_reference(tmp_path):
    typed = record(1, "input_text", {"text": "hello"})
    secret = record(2, "type_secret", {"name": "demo", "field": "password", "package": "app"})
    state = AgentState(goal="login", actions=[typed, secret])
    macro_path = tmp_path / "macros/run/macro.json"
    await RunRecorder(tmp_path / "run", macro_path=macro_path).finish(state)
    device = MemoryDevice()
    player = MacroPlayer(device)
    macro = player.load_macro_from_file(macro_path)
    assert await player.replay_macro(macro, max_steps=1)
    assert device.inputs == ["hello"]
    assert macro["actions"][1] == {
        "action_type": "type_secret",
        **secret.args,
        "pre_state": macro["actions"][0]["pre_state"],
    }


async def test_future_typed_literal_absent_from_every_checkpoint(tmp_path):
    recorder = RunRecorder(tmp_path, macro_path=tmp_path / "macros/run/macro.json")
    marker = "fixture-later-input"
    for step, name, args in [
        (1, "notes.save", {marker: marker}),
        (2, "input_text", {"text": marker}),
    ]:
        snapshot = StepSnapshot(
            goal=marker,
            step=step,
            ui=UIState([], "", "", {}, 0, 0),
            device_context=[],
        )
        await recorder.on_step(record(step, name, args, summary=marker, before=snapshot))
        assert marker not in (tmp_path / f"checkpoints/{step:04}.json").read_text()
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert [p.stem for p in checkpoints] == ["0001", "0002"]
    assert all(marker not in p.read_text() for p in checkpoints)


async def test_action_record_requires_observation():
    with pytest.raises(ValueError, match="before"):
        ActionRecord(step=1, name="tap", args={"x": 3, "y": 4}, ok=True, summary="ok")


async def test_expanded_credential_is_rejected_before_publishing_anything(tmp_path):
    action = record(1, "type_secret", {"name": "demo", "text": "not-a-reference"})
    state = AgentState(goal="login", actions=[action])
    macro_path = tmp_path / "macros/run/macro.json"
    recorder = RunRecorder(tmp_path / "run", macro_path=macro_path)
    with pytest.raises(ValueError, match="vault reference"):
        await recorder.on_step(action)
    with pytest.raises(ValueError, match="vault reference"):
        await recorder.finish(state)
    assert not (tmp_path / "run").exists()
    assert not macro_path.exists()

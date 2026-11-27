import json
from types import SimpleNamespace

import pytest
from mobilerun.core.devices import RecordingDevice
from mobilerun.core.runs import recording_scope

from mobilerun_agents.agent.operators.recording import resolve_recording
from mobilerun_agents.config.loader import ConfigLoader
from tests.unit.device_fixtures import MemoryDevice


@pytest.mark.parametrize(
    ("explicit", "env", "yaml", "expected"),
    [
        (True, "false", False, True),
        (False, "true", True, False),
        (None, "true", False, True),
        (None, "false", True, False),
        (None, None, True, True),
        (None, None, False, False),
    ],
)
def test_recording_precedence(monkeypatch, tmp_path, explicit, env, yaml, expected):
    monkeypatch.delenv("MOBILERUN_AGENT_RECORDING_ENABLED", raising=False)
    if env is not None:
        monkeypatch.setenv("MOBILERUN_AGENT_RECORDING_ENABLED", env)
    phone = MemoryDevice()
    path = tmp_path / "agents.yaml"
    path.write_text(f"recording:\n  enabled: {str(yaml).lower()}\n")
    config = ConfigLoader.load(str(path))
    device, recorder, owned = resolve_recording(phone, explicit, config)
    assert owned is expected
    if expected:
        assert device is recorder and device.recording_device() is recorder
    else:
        assert device is phone and recorder is None


@pytest.mark.parametrize("stored", [True, False])
def test_saving_or_updating_config_does_not_persist_env_override(monkeypatch, tmp_path, stored):
    path = tmp_path / "agents.yaml"
    path.write_text(f"recording:\n  enabled: {str(stored).lower()}\n")
    monkeypatch.setattr(ConfigLoader, "get_user_config_path", lambda: path)
    monkeypatch.setenv("MOBILERUN_AGENT_RECORDING_ENABLED", str(not stored).lower())
    config = ConfigLoader.load()
    assert config.recording.enabled is not stored
    ConfigLoader.save(config)
    ConfigLoader.update(lambda value: setattr(value.agent, "max_steps", 9))
    monkeypatch.delenv("MOBILERUN_AGENT_RECORDING_ENABLED")
    reloaded = ConfigLoader.load()
    assert reloaded.recording.enabled is stored and reloaded.agent.max_steps == 9


def test_resolution_reuses_existing_recorder_and_respects_parent_scope():
    recorder = RecordingDevice(MemoryDevice())
    config = SimpleNamespace(recording=SimpleNamespace(enabled=True))
    assert resolve_recording(recorder, True, config) == (recorder, recorder, False)
    with recording_scope(enabled=False):
        assert resolve_recording(recorder, True, config) == (recorder, None, False)


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_auto_recorder_records_action_without_observing_device(monkeypatch, tmp_path):
    class Phone(MemoryDevice):
        def __init__(self):
            super().__init__()
            self._set_state_provider(SimpleNamespace(get_final_state=self.get_final_state))

        async def get_final_state(self):
            pytest.fail("Recording an action must not fetch UI state")

        async def screenshot(self, *args, **kwargs):
            pytest.fail("Recording an action must not take a screenshot")

    monkeypatch.setenv("MOBILERUN_RUNS_DIR", str(tmp_path))
    phone = Phone()
    config = SimpleNamespace(recording=SimpleNamespace(enabled=True))
    device, recorder, owned = resolve_recording(phone, None, config)
    assert device is recorder and owned
    await device.input_text("fixture")
    folder = recorder.run_dir
    assert folder is not None
    assert {path.name for path in folder.iterdir()} == {"run.json", "steps.jsonl"}
    step = json.loads((folder / "steps.jsonl").read_text())
    assert step["action"] == "input_text"
    assert step["args"]["text"] == "fixture"
    assert step["ok"] is True
    assert step["started_at"] <= step["finished_at"]
    assert not any(step.get(key) for key in ("before", "after", "screens"))


@pytest.mark.parametrize("initial_env", [None, "true", "false"])
def test_loader_does_not_reapply_env_to_context_override(monkeypatch, tmp_path, initial_env):
    path = tmp_path / "agents.yaml"
    path.write_text("recording:\n  enabled: true\n")
    monkeypatch.delenv("MOBILERUN_AGENT_RECORDING_ENABLED", raising=False)
    if initial_env is not None:
        monkeypatch.setenv("MOBILERUN_AGENT_RECORDING_ENABLED", initial_env)
    resolved = ConfigLoader.load(str(path))
    monkeypatch.setenv(
        "MOBILERUN_AGENT_RECORDING_ENABLED", str(not resolved.recording.enabled).lower()
    )
    with ConfigLoader.override(resolved):
        assert ConfigLoader.load() is resolved
        assert ConfigLoader.load(str(path)).recording.enabled is not resolved.recording.enabled

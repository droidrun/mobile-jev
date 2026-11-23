import json

import pytest
from mobilerun.macro.models import (
    MACRO_SCHEMA_VERSION,
    UNSUPPORTED_SCHEMA_MESSAGE,
)

from mobilerun.agents.trajectory.trajectory import Trajectory
from mobilerun.agents.trajectory.writer import TrajectoryWriter


def test_macro_loader_rejects_files_without_v2_schema(tmp_path):
    macro_path = tmp_path / "macro.json"
    macro_path.write_text(json.dumps({"version": "1.0", "actions": [{"action_type": "tap"}]}))

    with pytest.raises(ValueError, match=UNSUPPORTED_SCHEMA_MESSAGE):
        Trajectory.load_macro_sequence(str(macro_path))


def test_trajectory_writer_writes_only_v2_macro_files(tmp_path):
    trajectory = Trajectory(goal="open settings", base_path=str(tmp_path))
    trajectory.macro = [
        {
            "action_type": "wait",
            "duration": 1.5,
            "pre_state": {"nodes": []},
            "recorded_at_ms": 1000,
            "elapsed_since_previous_ms": 0,
        }
    ]

    writer = TrajectoryWriter()
    job = writer._create_macro_job(
        list(trajectory.macro), trajectory, trajectory.trajectory_folder.name, "final"
    )

    assert job is not None
    macro_data = json.loads(job.serialized_macro)
    assert macro_data["macro_schema_version"] == MACRO_SCHEMA_VERSION
    assert macro_data["version"] == MACRO_SCHEMA_VERSION
    assert macro_data["total_actions"] == 1
    assert macro_data["actions"][0]["action_type"] == "wait"


def test_trajectory_writer_strips_target_hint_from_macro_actions(tmp_path):
    trajectory = Trajectory(goal="tap settings", base_path=str(tmp_path))
    trajectory.macro = [
        {
            "action_type": "tap",
            "x": 10,
            "y": 20,
            "pre_state": {"nodes": []},
            "target_hint": {"text": "Settings"},
            "recorded_at_ms": 1000,
            "elapsed_since_previous_ms": 0,
        }
    ]

    writer = TrajectoryWriter()
    job = writer._create_macro_job(
        list(trajectory.macro), trajectory, trajectory.trajectory_folder.name, "final"
    )

    assert job is not None
    macro_data = json.loads(job.serialized_macro)
    assert "target_hint" not in macro_data["actions"][0]
    assert macro_data["actions"][0]["x"] == 10
    assert macro_data["actions"][0]["y"] == 20

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from llama_index.core.workflow import Event

from mobilerun.agents.trajectory.writer import TrajectoryWriter


def test_dynamic_workflow_tokens_are_serialized_without_attribute_probing(tmp_path):
    path = Path(__import__("mobilerun.agents.trajectory.writer", fromlist=["__file__"]).__file__)
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hasattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "tokens"
        for node in ast.walk(tree)
    )
    writer = TrajectoryWriter.__new__(TrajectoryWriter)
    job = writer._create_events_job(
        [Event(tokens={"input": 3})],
        SimpleNamespace(trajectory_folder=tmp_path),
        "id",
        "final",
    )
    assert json.loads(job.serialized_events)[0]["tokens"] == {"input": 3}

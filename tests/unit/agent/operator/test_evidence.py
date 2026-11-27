import json
from unittest.mock import patch

import pytest

from mobilerun_agents.agent.operators.evidence import AgentEntry, AgentEvidence


def test_evidence_appends_across_consumers_without_losing_step_links(tmp_path):
    entry = AgentEntry(kind="thought", text="Use the search tool", step_ids=[2, 3])
    AgentEvidence(tmp_path).append(entry)
    AgentEvidence(tmp_path).append(AgentEntry(kind="tool", text="complete"))

    assert json.loads((tmp_path / "agent.json").read_text()) == [
        {"kind": "thought", "text": "Use the search tool", "step_ids": [2, 3]},
        {"kind": "tool", "text": "complete", "step_ids": []},
    ]


def test_failed_atomic_replace_preserves_existing_evidence(tmp_path):
    evidence = AgentEvidence(tmp_path)
    evidence.append(AgentEntry(kind="tool", text="tap", step_ids=[1]))
    path = tmp_path / "agent.json"
    before = path.read_bytes()
    contents = set(tmp_path.iterdir())
    with (
        patch("mobilerun_agents.agent.operators.evidence.os.replace", side_effect=OSError),
        pytest.raises(OSError),
    ):
        evidence.append(AgentEntry(kind="tool", text="input_text", step_ids=[2]))
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == contents

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_metadata():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert data["name"] == "mobilerun-agents"
    assert data["requires-python"] == ">=3.11,<3.14"
    assert "mobilerun>=0.7.4" in data["dependencies"]
    assert not (ROOT / "package.json").exists()

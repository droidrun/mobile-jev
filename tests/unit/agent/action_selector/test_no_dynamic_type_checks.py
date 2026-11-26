import re
from pathlib import Path


def test_no_dynamic_type_checks():
    root = Path(__file__).resolve().parents[4] / "src/mobilerun_agents/agent/action_selector"
    assert root.is_dir()
    violations = [
        f"{path.relative_to(root)}:{number}: {line.strip()}"
        for path in root.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\b(?:isinstance|getattr|hasattr|cast)\s*\(", line)
    ]
    assert not violations, "\n".join(violations)

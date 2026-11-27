"""Library modules must not call the built-in print function."""

import ast
from pathlib import Path


def test_library_does_not_call_print():
    root = Path(__import__("mobilerun_agents", fromlist=["__file__"]).__file__).parent
    violations = []
    assert any(root.rglob("*.py"))
    for path in root.rglob("*.py"):
        if "cli" in path.relative_to(root).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations

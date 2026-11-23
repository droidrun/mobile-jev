"""Library modules must not call the built-in print function."""

import ast
from pathlib import Path


def test_library_does_not_call_print():
    root = Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
    violations = []
    assert any(root.rglob("*.py"))
    for path in root.rglob("*.py"):
        # Opt-in checkout handlers deliberately print payment instructions.
        if path == root / "utils" / "checkout_handlers.py":
            continue
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


def test_trajectory_summary_logs_instead_of_printing(monkeypatch, capsys, caplog):
    import logging

    from mobilerun.agents.trajectory.trajectory import Trajectory

    monkeypatch.setattr(Trajectory, "load_macro_sequence", lambda _: {"actions": []})
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.INFO, logger="mobilerun"):
        Trajectory.print_macro_summary("fake")
    assert not capsys.readouterr().out
    assert any("Macro Summary" in message for message in caplog.messages)

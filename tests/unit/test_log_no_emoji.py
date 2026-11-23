"""Library log messages should not contain decorative emoji."""

import ast
from pathlib import Path


def test_library_log_messages_have_no_emoji():
    root = Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
    violations = []
    for path in root.rglob("*.py"):
        if path == root / "logs" / "actions.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {
                "debug",
                "info",
                "warning",
                "error",
                "exception",
                "critical",
            }:
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {
                "logger",
                "log",
                "logging",
            }:
                continue
            for arg in node.args[:1]:
                for part in ast.walk(arg):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        if any(
                            0x1F300 <= ord(c) <= 0x1FAFF
                            or 0x2300 <= ord(c) <= 0x23FF
                            or 0x2600 <= ord(c) <= 0x27BF
                            for c in part.value
                        ):
                            violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations


def test_trajectory_internal_path_logs_only_at_debug(tmp_path, caplog, monkeypatch):
    import logging

    from mobilerun.agents.trajectory.trajectory import Trajectory

    logger = logging.getLogger("mobilerun")
    root_handlers = logging.getLogger().handlers
    monkeypatch.setattr(logger, "handlers", [h for h in logger.handlers if h not in root_handlers])
    monkeypatch.setattr(logger, "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="mobilerun"):
        Trajectory(base_path=str(tmp_path))
    records = [record for record in caplog.records if "Trajectory folder" in record.message]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG


def test_telemetry_log_constant_has_no_emoji():
    from mobilerun.agents.telemetry.tracker import TELEMETRY_DISABLED_MESSAGE

    assert all(
        not (
            0x1F300 <= ord(c) <= 0x1FAFF or 0x2300 <= ord(c) <= 0x23FF or 0x2600 <= ord(c) <= 0x27BF
        )
        for c in TELEMETRY_DISABLED_MESSAGE
    )


def test_verbose_agent_diagnostics_are_debug_level():
    root = Path(__import__("mobilerun.agents", fromlist=["__file__"]).__file__).parent
    cases = [
        ("runner.py", "Vision settings"),
        ("workflows/fast_agent/fast_agent.py", "Tool results:"),
    ]
    for relative, message in cases:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        matches = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"info", "debug"}
            and node.args
            and message
            in ast.get_source_segment((root / relative).read_text(encoding="utf-8"), node.args[0])
        ]
        assert len(matches) == 1 and matches[0].func.attr == "debug"

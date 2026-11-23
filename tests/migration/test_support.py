"""Static migration gates for the support surfaces before peers are installed."""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from signature_helper import canonical_signature

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "src/mobilerun/agents"
MANIFEST = json.loads((ROOT / "scripts/source_manifest.json").read_text())
MAP = json.loads((ROOT / "scripts/module_map.json").read_text())
AREAS = {"config", "cli", "clients", "context", "mcp", "errors", "logs", "telemetry"}
EXTRA = {"agent/utils/errors.py", "agent/utils/tracing_setup.py", "logging_setup.py", "runner.py"}


def owned(path):
    return path in EXTRA or path.split("/")[0] in AREAS


def target(path):
    parts = path.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts.pop()
    old = ".".join(["mobilerun", "agents", *parts])
    mapped = MAP[old].removeprefix("mobilerun.agents.").replace(".", "/")
    return (
        DEST / mapped / "__init__.py" if path.endswith("/__init__.py") else DEST / (mapped + ".py")
    )


def test_all_support_modules_and_signatures_are_preserved():
    for path, data in MANIFEST.items():
        if not path.endswith(".py") or not owned(path):
            continue
        dest = target(path)
        assert dest.is_file(), f"missing {path} -> {dest}"
        if path.endswith("/__init__.py"):
            continue  # package exports may be made lazy
        tree = ast.parse(dest.read_text())
        actual = Counter(
            (n.name, canonical_signature(ast.dump(n.args, include_attributes=False)))
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        expected = Counter(
            (key.rsplit(":", 1)[0], canonical_signature(value))
            for key, value in data["signatures"].items()
        )
        assert actual == expected, path


def test_assets_are_identical():
    assets = {p: d for p, d in MANIFEST.items() if p.startswith("assets/")}
    assert assets
    for path, data in assets.items():
        file = DEST / path
        assert file.is_file(), path
        assert hashlib.sha256(file.read_bytes()).hexdigest() == data["sha256"], path


def test_cli_dispatch_files_and_config_inheritance():
    for path in ("cli/agent/group.py", "cli/agent/run.py", "cli/agent/llm/configure.py"):
        assert (DEST / path).is_file(), path
    tree = ast.parse((DEST / "config/user_config.py").read_text())
    assert any(
        isinstance(n, ast.ClassDef)
        and n.name == "MobileConfig"
        and any(isinstance(b, ast.Name) and b.id == "BaseMobileConfig" for b in n.bases)
        for n in ast.walk(tree)
    )


def test_exception_implementations_are_unique():
    classes = {}
    for path in (DEST / "errors").glob("*.py"):
        if path.name == "timeout.py":
            continue
        for n in ast.parse(path.read_text()).body:
            if isinstance(n, ast.ClassDef):
                classes.setdefault(n.name, []).append(path.name)
    assert classes.get("AgentTimeout") == ["agent_timeout.py"]
    assert all(len(files) == 1 for files in classes.values())


def test_telemetry_package_exports_are_lazy():
    tree = ast.parse((DEST / "telemetry/__init__.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body)
    assert not any(
        isinstance(n, ast.ImportFrom)
        and n.module
        and n.module.startswith("mobilerun.agents.telemetry")
        for n in tree.body
    )


def test_required_support_api_definitions():
    expected = {
        "runner.py": {"AgentRunResult", "run_agent"},
        "config/user_config.py": {"MobileConfig", "AgentConfig", "LLMProfile"},
        "logs/error_descriptions.py": {"HtmlErrorPageFilter", "describe_error"},
        "mcp/adapter.py": {"mcp_to_mobilerun_tools"},
        "mcp/client.py": {"MCPClientManager", "MCPToolInfo"},
        "mcp/config.py": {"MCPConfig", "MCPServerConfig"},
    }
    for path, names in expected.items():
        tree = ast.parse((DEST / path).read_text())
        definitions = {
            n.name
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert names <= definitions, path

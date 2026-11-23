"""Static extraction checks against the frozen source inventory."""

import ast
import json
from collections import Counter
from pathlib import Path

import pytest
from signature_helper import canonical_signature

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "scripts/source_manifest.json").read_text())
MAPPING = json.loads((ROOT / "scripts/module_map.json").read_text())
PREFIX = "mobilerun.agents."


def workflow_files():
    for source in MANIFEST:
        if not source.startswith("agent/") or not source.endswith(".py"):
            continue
        parts = Path(source).with_suffix("").parts
        module = PREFIX + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        target = MAPPING.get(module, "")
        if target == PREFIX + "workflows" or target.startswith(PREFIX + "workflows."):
            relative = target.removeprefix(PREFIX).replace(".", "/")
            yield source, ROOT / "src/mobilerun/agents" / (
                relative + ("/__init__.py" if parts[-1] == "__init__" else ".py")
            )


FILES = list(workflow_files())


@pytest.mark.parametrize("source,target", FILES, ids=[x[0] for x in FILES])
def test_workflow_source_has_parseable_destination(source, target):
    assert target.is_file(), f"{source} missing destination {target}"
    ast.parse(target.read_text())


@pytest.mark.parametrize("source,target", FILES, ids=[x[0] for x in FILES])
def test_workflow_signatures_match_source_manifest(source, target):
    if target.name == "__init__.py":
        pytest.skip("package exports may become lazy")
    tree = ast.parse(target.read_text())
    expected = Counter(
        (key.rsplit(":", 1)[0], canonical_signature(value))
        for key, value in MANIFEST[source]["signatures"].items()
    )
    actual = Counter(
        (node.name, canonical_signature(ast.dump(node.args, include_attributes=False)))
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert actual == expected


@pytest.mark.parametrize("source,target", FILES, ids=[x[0] for x in FILES])
def test_workflows_do_not_import_legacy_agent_modules(source, target):
    tree = ast.parse(target.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mobilerun.agents.agent")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("mobilerun.agents.agent") for alias in node.names)


def test_keyword_option_types_match_frozen_source():
    """The TypedDict fields and inheritance survive the move exactly."""
    import hashlib

    target = ROOT / "src/mobilerun/agents/workflows/options.py"
    assert (
        hashlib.sha256(target.read_bytes()).hexdigest()
        == MANIFEST["agent/workflow_options.py"]["sha256"]
    )


def test_external_plugin_discovery_and_loading_from_canonical_package(tmp_path, monkeypatch):
    import importlib
    import sys
    import sysconfig

    from mobilerun.agents.workflows import external
    from mobilerun.agents.workflows.external import loader

    assert Path(loader.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")))
    name = "migration_fake_plugin"
    module_name = f"{external.__name__}.{name}"
    plugin = tmp_path / f"{name}.py"
    plugin.write_text("DEFAULT_CONFIG = {'mode': 'fake'}\nasync def run():\n    return 'ok'\n")
    monkeypatch.setattr(loader, "_EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(external, "__path__", [str(tmp_path)])
    importlib.invalidate_caches()
    try:
        assert loader.list_agents() == [name]
        loaded = loader.load_agent(name)
        assert loaded is not None
        assert loaded["config"] == {"mode": "fake"}
        assert loaded["run"].__module__ == module_name
    finally:
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()


def test_implementation_contains_no_legacy_agent_paths():
    root = ROOT / "src/mobilerun/agents"
    excluded = {"operator.py", "logging_setup.py", "errors/timeout.py"}
    needles = (b"mobilerun.agents.agent.", b"mobilerun/agents/agent/")
    violations = []
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in {"agent", "agent_macro"} or relative.as_posix() in excluded:
            continue
        contents = path.read_bytes()
        if any(needle in contents for needle in needles):
            violations.append(relative.as_posix())
    assert not violations, f"Legacy paths found in: {violations}"

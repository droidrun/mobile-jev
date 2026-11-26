"""Enforce the dependency direction of the agents package, including lazy imports."""

import ast
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "mobilerun_agents"
PREFIX = "mobilerun_agents"
LAYERS = {
    "agent.operators": 6,
    "agent.workflows.core": 2,
    "agent.run": 3,
    "agent.workflows": 5,
    **dict.fromkeys(
        (
            "agent.action_selector",
            "agent.planner",
            "agent.tools",
            "agent.memory",
            "agent.checkpoints",
            "agent.analytics",
            "agent.logs",
        ),
        4,
    ),
    "agent.auth": 1,
    "config": 0,
    "errors": 0,
    "cli": 8,
    "__init__": 8,
    "macro_agent": 7,
    "agent": 6,
}
TOOL_LAYERS = {
    "agent.tools.helpers": 0,
    "agent.tools.base": 1,
    "agent.tools.atomic": 2,
    "agent.tools.registry": 3,
    "agent.tools": 4,
}


def _under(name, package):
    return name == package or name.startswith(package + ".")


def _new_layer(name):
    return next((level for pkg, level in LAYERS.items() if _under(name, pkg)), None)


def _module(path):
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((PREFIX, *parts))


def _imports(path):
    module = _module(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    tree = ast.parse(path.read_text(), filename=str(path))
    # Run holds service objects but does not import those services at runtime.
    type_only = (
        {
            descendant
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            for statement in node.body
            for descendant in ast.walk(statement)
        }
        if module == f"{PREFIX}.agent.run"
        else set()
    )
    for node in ast.walk(tree):
        if node in type_only:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = ".".join(part for part in (base, node.module) if part)
            else:
                target = node.module or ""
            for alias in node.names:
                yield node.lineno, f"{target}.{alias.name}"
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith(".")
        ):
            relative_name = node.args[0].value
            package_name = (
                node.args[1].id
                if len(node.args) > 1 and isinstance(node.args[1], ast.Name)
                else package
            )
            if package_name == "__package__":
                package_name = package
            yield node.lineno, resolve_name(relative_name, package_name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Lazy public export maps are imports too.
            if node.value.startswith(PREFIX + "."):
                yield node.lineno, node.value


def test_import_layers():
    violations = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        origin = (
            _module(path).removeprefix(PREFIX + ".") if path != ROOT / "__init__.py" else "__init__"
        )
        source_layer = _new_layer(origin)
        if source_layer is None:
            violations.append(f"{relative}: unclassified source {origin}")
            continue
        for line, target in _imports(path):
            if target.startswith(PREFIX + "."):
                name = target.removeprefix(PREFIX + ".")
                level = _new_layer(name)
                source_tool = next((n for pkg, n in TOOL_LAYERS.items() if _under(origin, pkg)), -1)
                target_tool = next((n for pkg, n in TOOL_LAYERS.items() if _under(name, pkg)), -1)
                if level is None:
                    violations.append(f"{relative}:{line}: unclassified import {target}")
                elif level > source_layer or 0 <= source_tool < target_tool:
                    violations.append(f"{relative}:{line}: upward import {target}")
    assert not violations, "\n" + "\n".join(violations)


def test_external_plugins_have_no_workflow_module():
    module = "mobilerun_agents.agent.workflows.external"
    with pytest.raises(ModuleNotFoundError) as exc:
        import_module(module)
    assert exc.value.name == module


@pytest.mark.parametrize(
    "source,code,message",
    [
        ("agent/workflows/core/fake.py", "import mobilerun_agents.agent.memory", "upward import"),
        ("agent/analytics/fake.py", "from ..workflows.events import Act", "upward import"),
        ("agent/workflows/fake.py", "from ..operators import AgentOperator", "upward import"),
        ("agent/auth/fake.py", "from ..workflows.core import state", "upward import"),
        ("agent/workflows/core/fake.py", "from ...run import Run", "upward import"),
        ("agent/run.py", "from .memory.snapshot import StepSnapshot", "upward import"),
        ("agent/tools/atomic/schema.py", "from ..registry import ToolRegistry", "upward import"),
        (
            "agent/tools/atomic/fake.py",
            'from importlib import import_module\nimport_module("..registry", __package__)',
            "upward import",
        ),
        ("agent/tools/helpers/elements.py", "from ..atomic import AtomicTool", "upward import"),
        ("agent/tools/helpers/elements.py", "from ..base import Tool", "upward import"),
        (
            "agent/action_selector/fake.py",
            "import mobilerun_agents.providers",
            "unclassified import",
        ),
        ("config/fake.py", "import mobilerun_agents.agent.auth", "upward import"),
        (
            "agent/tools/fake.py",
            "from mobilerun_agents.providers import UsageResult",
            "unclassified import",
        ),
        (
            "macro_agent/fake.py",
            "from mobilerun_agents.tools import actions",
            "unclassified import",
        ),
    ],
)
def test_layer_guard_rejects_hidden_and_direct_upward_imports(
    tmp_path, monkeypatch, source, code, message
):
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text(code)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    with pytest.raises(AssertionError, match=message):
        test_import_layers()


def test_layer_guard_accepts_downward_relative_imports(tmp_path, monkeypatch):
    path = tmp_path / "agent/workflows/example.py"
    path.parent.mkdir(parents=True)
    path.write_text("from .core.state import AgentState")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    test_import_layers()


def test_layer_guard_accepts_v3_package_imports(tmp_path, monkeypatch):
    path = tmp_path / "agent/analytics/consumer.py"
    path.parent.mkdir(parents=True)
    path.write_text("from mobilerun_agents.agent import auth\nfrom ..workflows import core")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    test_import_layers()


def test_workflow_can_import_planner_service(tmp_path, monkeypatch):
    path = tmp_path / "agent/workflows/example.py"
    path.parent.mkdir(parents=True)
    path.write_text("from ..planner import Planner")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    test_import_layers()


@pytest.mark.parametrize(
    "source,code",
    [
        ("agent/tools/registry.py", "from .atomic import AtomicTool"),
        ("agent/tools/atomic/schema.py", "from ..helpers import check_point"),
        ("agent/tools/helpers/coordinates.py", "from ...run import Run"),
        ("agent/tools/helpers/elements.py", "from ...workflows.core import state"),
        ("agent/action_selector/base.py", "from ..tools.helpers import coordinate_pairs"),
        ("agent/run.py", "from .workflows.core import state"),
    ],
)
def test_tool_and_run_layers_accept_downward_imports(tmp_path, monkeypatch, source, code):
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    path.write_text(code)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    test_import_layers()

"""Static checks for the tools, trajectory and utilities extraction."""

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "scripts/source_manifest.json").read_text())
MAPPING = json.loads((ROOT / "scripts/module_map.json").read_text())
PREFIX = "mobilerun.agents."
OWNED = ("tools", "trajectory", "utils")


def mapped_files():
    for source in MANIFEST:
        if not source.endswith(".py"):
            continue
        parts = Path(source).with_suffix("").parts
        module = PREFIX + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        target = MAPPING.get(module, "").removeprefix(PREFIX)
        if target.split(".")[0] not in OWNED:
            continue
        suffix = "/__init__.py" if parts[-1] == "__init__" else ".py"
        yield source, ROOT / "src/mobilerun/agents" / (target.replace(".", "/") + suffix)


FILES = list(mapped_files())


@pytest.mark.parametrize("source,target", FILES, ids=[x[0] for x in FILES])
def test_mapped_files_exist_and_parse(source, target):
    assert target.is_file(), f"{source} missing destination {target}"
    ast.parse(target.read_text())


@pytest.mark.parametrize("source,target", FILES, ids=[x[0] for x in FILES])
def test_callable_signatures_preserved(source, target):
    if target.name == "__init__.py":
        pytest.skip("initializer exports are merged or made lazy")
    expected = Counter(
        (name.rsplit(":", 1)[0], signature)
        for name, signature in MANIFEST[source]["signatures"].items()
    )
    actual = Counter(
        (
            node.name,
            re.sub(
                r", ctx=(?:Load|Store|Del)\(\)", "", ast.dump(node.args, include_attributes=False)
            ),
        )
        for node in ast.walk(ast.parse(target.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert actual == expected


def test_merged_utility_exports_preserve_legacy_names():
    tree = ast.parse((ROOT / "src/mobilerun/agents/utils/__init__.py").read_text())
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert set(ast.literal_eval(assignments["__all__"])) == {
        "to_chat_messages",
        "has_content",
        "filter_empty_messages",
        "limit_history",
        "PromptResolver",
        "build_tool_registry",
        "Trajectory",
    }
    exports = ast.literal_eval(assignments["_EXPORTS"])
    assert set(exports) == set(ast.literal_eval(assignments["__all__"]))
    assert exports["build_tool_registry"] == "mobilerun.agents.tools.signatures"
    assert exports["Trajectory"] == "mobilerun.agents.trajectory.trajectory"
    assert exports["PromptResolver"] == "mobilerun.agents.utils.prompt_resolver"
    for name in ("to_chat_messages", "has_content", "filter_empty_messages", "limit_history"):
        assert exports[name] == "mobilerun.agents.utils.chat_utils"


def test_paths_resource_root_remains_agents_directory():
    path = ROOT / "src/mobilerun/agents/utils/paths.py"
    tree = ast.parse(path.read_text())
    resolver = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PathResolver"
    )
    method = next(
        n for n in resolver.body if isinstance(n, ast.FunctionDef) and n.name == "get_project_root"
    )
    assert ast.unparse(method.body[0].value) == "Path(__file__).resolve().parents[1]"
    assert path.parents[1].name == "agents"

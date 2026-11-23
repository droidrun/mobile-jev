import ast
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "scripts/source_manifest.json").read_text())
MAP = json.loads((ROOT / "scripts/module_map.json").read_text())
PREFIX = "mobilerun.agents."
OWNED = {old: new for old, new in MAP.items() if new.startswith(PREFIX + "llm.") and old != new}


def path_for(module):
    relative = module.removeprefix(PREFIX).replace(".", "/")
    root = ROOT / "src/mobilerun/agents"
    return (
        root / relative / "__init__.py"
        if (root / relative / "__init__.py").exists()
        else root / (relative + ".py")
    )


def old_path(module):
    relative = module.removeprefix(PREFIX).replace(".", "/")
    candidate = relative + ".py"
    return candidate if candidate in MANIFEST else relative + "/__init__.py"


@pytest.mark.parametrize("old,new", sorted(OWNED.items()))
def test_copied_signatures_and_literals(old, new):
    entry = MANIFEST[old_path(old)]
    tree = ast.parse(path_for(new).read_text())
    signatures = Counter(
        ast.dump(n.args, include_attributes=False).replace(", ctx=Load()", "")
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    assert signatures == Counter(entry.get("signatures", {}).values())
    assignments = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            try:
                value = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = value
    for name, value in entry.get("literals", {}).items():
        assert assignments[name] == value


def test_oauth_wrapper_inventory():
    for name in (
        "anthropic_oauth_llm",
        "gemini_oauth_code_assist_llm",
        "grok_oauth_llm",
        "openai_oauth_llm",
    ):
        assert (ROOT / "src/mobilerun/agents/llm/oauth" / (name + ".py")).is_file()


def test_dynamic_provider_modules_are_canonical():
    picker = (ROOT / "src/mobilerun/agents/llm/llm_picker.py").read_text()
    assert "mobilerun.agents.agent.providers" not in picker
    assert "mobilerun.agents.llm.providers" in picker


def test_llm_exports_are_lazy():
    tree = ast.parse((ROOT / "src/mobilerun/agents/llm/__init__.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body)
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "llm_picker" in ast.unparse(n)
        for n in tree.body
    )
    assert "load_llm" in ast.unparse(tree)


def test_lazy_exports_resolve_correct_modules(monkeypatch):
    import runpy
    from types import SimpleNamespace

    namespace = runpy.run_path(str(ROOT / "src/mobilerun/agents/llm/__init__.py"))
    lookup = namespace["__getattr__"]
    loaded = []
    sentinel = object()

    def fake_import(name):
        loaded.append(name)
        return SimpleNamespace(load_llm=sentinel, load_agent_llms=sentinel)

    monkeypatch.setitem(lookup.__globals__, "import_module", fake_import)
    assert loaded == []
    assert lookup("load_llm") is sentinel
    assert lookup("load_agent_llms") is sentinel
    assert loaded == ["mobilerun.agents.llm.llm_picker", "mobilerun.agents.llm.llm_loader"]
    with pytest.raises(AttributeError):
        lookup("missing")


def test_retry_status_codes_unchanged():
    tree = ast.parse((ROOT / "src/mobilerun/agents/llm/inference.py").read_text())
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_RETRYABLE_HTTP_CLIENT_STATUS_CODES"
            for t in n.targets
        )
    )
    assert ast.literal_eval(assignment.value) == {408, 409, 425, 429}

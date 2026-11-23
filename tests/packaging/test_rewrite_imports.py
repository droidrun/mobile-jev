import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_rewrite(tmp_path):
    script = ROOT / "scripts/rewrite_imports.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("rewrite_imports", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mapping = {
        "mobilerun.agents.agent": "mobilerun.agents.workflows",
        "mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo",
    }
    text = 'from mobilerun.agents.agent import foo\nfrom .foo import Thing\nx = "mobilerun.agents.agent.foo.Thing"\ny = "ordinary mobilerun.agents.agent.foo prose"\n'
    result = mod.rewrite(text, "mobilerun.agents.agent.bar", False, mapping)
    assert "from mobilerun.agents.tools import foo" in result
    assert "from mobilerun.agents.tools.foo import Thing" in result
    assert '"mobilerun.agents.tools.foo.Thing"' in result
    assert "ordinary mobilerun.agents.agent.foo prose" in result


def load_rewriter():
    spec = importlib.util.spec_from_file_location("rewriter", ROOT / "scripts/rewrite_imports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unicode_multiline_and_dynamic_imports():
    mod = load_rewriter()
    mapping = {
        "mobilerun.agents.agent": "mobilerun.agents.workflows",
        "mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo",
        "mobilerun.agents.agent.bar": "mobilerun.agents.llm.bar",
    }
    source = """# café
from mobilerun.agents.agent import (
    foo as f,
    Thing,
    bar,
)
from .foo import Thing as Alias
import importlib
importlib.import_module("mobilerun.agents.agent.foo")
patch("mobilerun.agents.agent.foo.Thing")
_EXPORTS = {"Foo": ("mobilerun.agents.agent.foo", "Foo")}
prose = "Use mobilerun.agents.agent.foo for this"
"""
    result = mod.rewrite(source, "mobilerun.agents.agent.test", False, mapping)
    compile(result, "unicode.py", "exec")
    assert "from mobilerun.agents.tools import (" in result and "foo as f" in result
    assert "from mobilerun.agents.workflows import Thing" in result
    assert "from mobilerun.agents.llm import bar" in result
    assert "from mobilerun.agents.tools.foo import Thing as Alias" in result
    assert result.count('"mobilerun.agents.tools.foo"') == 2
    assert 'patch("mobilerun.agents.tools.foo.Thing")' in result
    assert '"Use mobilerun.agents.agent.foo for this"' in result


def test_check_finds_unmapped_paths_and_excludes_compat(tmp_path):
    import json
    import subprocess
    import sys

    mapping = {
        "mobilerun.agents.agent": "mobilerun.agents.workflows",
        "mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo",
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps(mapping))
    root = tmp_path / "mobilerun/agents"
    root.mkdir(parents=True)
    (root / "one.py").write_text("import mobilerun.agents.agent.missing\n")
    (root / "two.py").write_text("from mobilerun.agents.agent.foo import Thing\n")
    compat = root / "agent"
    compat.mkdir()
    (compat / "compat.py").write_text("import mobilerun.agents.agent.foo\n")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/rewrite_imports.py"),
        "--root",
        str(root),
        "--map",
        str(path),
        "--check",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0
    assert "one.py" in result.stdout and "two.py" in result.stdout
    assert "compat.py" not in result.stdout


def test_real_source_files_compile_and_check(tmp_path):
    import json
    import shutil
    import subprocess
    import sys

    mapping = json.loads((ROOT / "scripts/module_map.json").read_text())
    for old in ("agent/action_context.py", "agent/action_result.py", "agent/usage.py"):
        module = "mobilerun.agents." + old.removesuffix(".py").replace("/", ".")
        new = mapping[module].removeprefix("mobilerun.agents.").replace(".", "/") + ".py"
        target = tmp_path / "mobilerun/agents" / new
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "tests/packaging/fixtures/source" / (old + ".txt"), target)
    root = tmp_path / "mobilerun/agents"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/rewrite_imports.py"),
        "--root",
        str(root),
        "--map",
        str(ROOT / "scripts/module_map.json"),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run(cmd + ["--check"], check=True)
    for path in root.rglob("*.py"):
        text = path.read_text()
        compile(text, str(path), "exec")
        assert "mobilerun.agents.agent." not in text


def test_check_allows_canonical_relative_imports(tmp_path):
    import json
    import subprocess
    import sys

    root = tmp_path / "mobilerun/agents"
    (root / "tools").mkdir(parents=True)
    (root / "tools/a.py").write_text("from .b import Thing\n")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"mobilerun.agents.agent.foo": "mobilerun.agents.tools.a"}))
    cmd = [
        sys.executable,
        str(ROOT / "scripts/rewrite_imports.py"),
        "--root",
        str(root),
        "--map",
        str(mapping),
        "--check",
    ]
    assert subprocess.run(cmd, capture_output=True).returncode == 0


def test_unicode_before_import_same_line():
    mod = load_rewriter()
    mapping = {"mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo"}
    result = mod.rewrite(
        "é = 1; from mobilerun.agents.agent.foo import Thing\n",
        "mobilerun.agents.x",
        False,
        mapping,
    )
    compile(result, "unicode.py", "exec")
    assert result == "é = 1; from mobilerun.agents.tools.foo import Thing\n"


def test_import_comments_and_qualified_references_preserved():
    mod = load_rewriter()
    mapping = {"mobilerun.agents.agent.action_context": "mobilerun.agents.tools.action_context"}
    source = "import mobilerun.agents.agent.action_context  # noqa: F401\nx = mobilerun.agents.agent.action_context.ActionContext\n"
    result = mod.rewrite(source, "mobilerun.agents.test", False, mapping)
    assert "# noqa: F401" in result
    assert "mobilerun.agents.tools.action_context.ActionContext" in result
    compile(result, "x.py", "exec")


def test_from_renamed_child_preserves_binding_and_comments():
    mod = load_rewriter()
    mapping = {
        "mobilerun.agents.agent": "mobilerun.agents.workflows",
        "mobilerun.agents.agent.foo": "mobilerun.agents.tools.bar",
    }
    source = "from mobilerun.agents.agent import (\n    foo,  # type: ignore\n)\nx = foo\n"
    result = mod.rewrite(source, "mobilerun.agents.test", False, mapping)
    assert "bar as foo" in result
    assert "# type: ignore" in result
    compile(result, "x.py", "exec")


def test_unsupported_qualified_assignment_fails_with_location():
    import pytest

    mod = load_rewriter()
    mapping = {"mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo"}
    source = "import mobilerun.agents.agent.foo\nmobilerun.agents.agent.foo.value = 1\n"
    with pytest.raises(ValueError, match="line 2"):
        mod.rewrite(source, "mobilerun.agents.test", False, mapping)


def test_split_import_with_comments_fails_with_file_and_line(tmp_path):
    import json
    import subprocess
    import sys

    root = tmp_path / "mobilerun/agents"
    root.mkdir(parents=True)
    target = root / "example.py"
    source = "from mobilerun.agents.agent import (\n    Thing,\n    foo,  # type: ignore[attr-defined]  # noqa: F401\n)\n"
    target.write_text(source)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "mobilerun.agents.agent": "mobilerun.agents.workflows",
                "mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo",
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/rewrite_imports.py"),
            "--root",
            str(root),
            "--map",
            str(mapping),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert str(target) in result.stderr and "line 1" in result.stderr
    assert "split import with comments; rewrite manually" in result.stderr
    assert target.read_text() == source


def test_split_import_without_comments_succeeds():
    mod = load_rewriter()
    mapping = {
        "mobilerun.agents.agent": "mobilerun.agents.workflows",
        "mobilerun.agents.agent.foo": "mobilerun.agents.tools.foo",
    }
    source = "from mobilerun.agents.agent import (\n    Thing,\n    foo,\n)\n"
    result = mod.rewrite(source, "mobilerun.agents.test", False, mapping)
    compile(result, "x.py", "exec")
    assert "from mobilerun.agents.tools import foo" in result

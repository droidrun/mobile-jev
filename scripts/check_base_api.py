"""Fail early when the installed base lacks modules imported by the add-on."""

import ast
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def required_modules(root):
    modules = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            modules.update(
                name
                for name in names
                if name.startswith("mobilerun.") and not name.startswith("mobilerun.agents")
            )
    return modules


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def check_modules(modules, available, version):
    missing = sorted(name for name in modules if not available(name))
    if missing:
        print(
            f"mobilerun {version} lacks required APIs: {', '.join(missing)}; "
            "the add-on needs the base rework release",
            file=sys.stderr,
        )
        return 1
    print(f"mobilerun {version}: required base modules available")
    return 0


if __name__ == "__main__":
    try:
        version = importlib.metadata.version("mobilerun")
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    sys.exit(
        check_modules(required_modules(ROOT / "src/mobilerun/agents"), module_available, version)
    )

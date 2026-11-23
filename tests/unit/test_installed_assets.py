"""Assets and CLI must work from an installed wheel outside either checkout."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EMPTY = ROOT / ".build" / "empty-cwd"


def test_installed_assets_resolve_and_render_outside_checkout():
    manifest = json.loads((ROOT / "scripts/source_manifest.json").read_text())
    assets = {
        name: item["sha256"]
        for name, item in manifest.items()
        if name.startswith(("assets/prompts/", "assets/app_cards/"))
    }
    assert assets
    report_path = ROOT / ".build" / "split-report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    # The requested venv may itself live inside ROOT; exclude its source tree,
    # rather than incorrectly rejecting the installed wheel for that reason.
    checkouts = [str(ROOT / "src")]
    if report.get("base_source"):
        checkouts.append(report["base_source"])
    EMPTY.mkdir(parents=True, exist_ok=True)
    script = """
import hashlib, json, sys, sysconfig
from pathlib import Path
from importlib.resources import files
from mobilerun.agents.utils.paths import PathResolver
from mobilerun.config.prompts import PromptLoader
import mobilerun.agents
purelib = Path(sysconfig.get_paths()['purelib']).resolve()
package = Path(mobilerun.agents.__file__).resolve()
assert package.is_relative_to(purelib), (package, purelib)
checkouts = [Path(path).resolve() for path in json.loads(sys.argv[2])]
assert all(not package.is_relative_to(checkout) for checkout in checkouts)
assets = json.loads(sys.argv[1])
for name, digest in assets.items():
    resolved = PathResolver.resolve(name, must_exist=True).resolve()
    resource = files('mobilerun.agents').joinpath(name)
    assert resolved == Path(str(resource)).resolve(), (name, resolved, resource)
    assert resolved.is_relative_to(purelib)
    assert all(not resolved.is_relative_to(checkout) for checkout in checkouts)
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == digest, name
for name in ('fast_agent/system.jinja2', 'manager/system.jinja2', 'executor/system.jinja2', 'agent_macro/extract.jinja2'):
    path = PathResolver.resolve('assets/prompts/' + name, must_exist=True)
    rendered = PromptLoader.render_template(path.read_text(), {
       'instruction': 'TASK_SENTINEL', 'goal': 'TASK_SENTINEL', 'platform': 'android', 'available_tools': [],
       'tool_descriptions': '', 'available_secrets': [], 'variables': {},
       'parallel_tools': False, 'vision': True, 'screenshot_only': False,
       'output_schema': None, 'atomic_actions': {}, 'actions': [], 'macro': [], 'steps': [],
    })
    assert rendered.strip(), name
    assert ('android' if name.startswith('fast_agent/') else 'TASK_SENTINEL') in rendered, name
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            json.dumps(assets),
            json.dumps(checkouts),
        ],
        cwd=EMPTY,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["agent", "--help"], ("run", "llm", "replay", "task")),
        (["agent", "llm", "--help"], ("configure", "login")),
        (["run", "--help"], ("--device",)),
    ],
)
def test_installed_cli_help_from_empty_cwd(args, expected, tmp_path):
    EMPTY.mkdir(parents=True, exist_ok=True)
    cli = Path(sys.executable).parent / "mobilerun"
    (tmp_path / "sitecustomize.py").write_text(
        "import socket, subprocess\n"
        "def blocked(*args, **kwargs): raise AssertionError('network or child process during help')\n"
        "socket.socket.connect = blocked\n"
        "socket.create_connection = blocked\n"
        "subprocess.Popen = blocked\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [str(cli), *args], cwd=EMPTY, capture_output=True, text=True, timeout=30, env=env
    )
    assert result.returncode == 0, result.stderr
    for name in expected:
        assert name in result.stdout

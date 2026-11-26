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
    import hashlib

    source = ROOT / "src/mobilerun_agents"
    assets = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source / "assets").rglob("*.jinja2")
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
from mobilerun_agents.config.paths import resolve_prompt_path
import mobilerun_agents
purelib = Path(sysconfig.get_paths()['purelib']).resolve()
package = Path(mobilerun_agents.__file__).resolve()
assert package.is_relative_to(purelib), (package, purelib)
checkouts = [Path(path).resolve() for path in json.loads(sys.argv[2])]
assert all(not package.is_relative_to(checkout) for checkout in checkouts)
assets = json.loads(sys.argv[1])
for name, digest in assets.items():
    resolved = resolve_prompt_path(name).resolve()
    resource = files('mobilerun_agents').joinpath(name)
    assert resolved == Path(str(resource)).resolve(), (name, resolved, resource)
    assert resolved.is_relative_to(purelib)
    assert all(not resolved.is_relative_to(checkout) for checkout in checkouts)
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == digest, name
from mobilerun_agents.config.prompts import render_selector_prompt
assert "example.settings" in render_selector_prompt("assets/prompts/tools/choose_app.jinja2", apps_list="example.settings", app_description="open settings")
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
        (["agent", "--help"], ("configure", "profiles", "providers", "run", "task")),
        (["agent", "providers", "add", "--help"], ("--api-key-stdin",)),
        (["agent", "profiles", "add", "--help"], ("--model", "--reasoning", "--min-confidence")),
        (["agent", "configure", "--help"], ("--selector", "--planner", "--executor", "--delegate")),
        (["agent", "run", "--help"], ("--steps", "--setup", "--vision")),
        (["macro-agent", "--help"], ("replay",)),
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

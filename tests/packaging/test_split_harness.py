import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_strip_only_selected_venv(tmp_path):
    script = ROOT / "scripts/prepare_split_env.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("split_harness", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = tmp_path / "venv"
    purelib = env / "lib/python3.13/site-packages"
    agents = purelib / "mobilerun/agents"
    agents.mkdir(parents=True)
    (agents / "sentinel").write_text("stale")
    mod.strip_agents(env, purelib)
    assert not agents.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError):
        mod.strip_agents(env, outside)


def test_wheel_metadata_dependencies(tmp_path):
    from zipfile import ZipFile

    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "mobilerun_agents-0.1.0.dist-info/METADATA",
            'Metadata-Version: 2.3\nName: mobilerun-agents\nVersion: 0.1.0\nRequires-Dist: mobilerun>=0.7.4\nRequires-Dist: alpha>=1\nRequires-Dist: beta>=2; extra == "dev"\nRequires-Dist: gamma; extra == "anthropic"\n\n',
        )
    assert mod.wheel_dependencies(wheel, ["dev"]) == ["alpha>=1", "beta>=2"]


def test_staging_ignores_secrets_but_not_python_modules():
    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ignored = mod.staging_ignore(
        "",
        [
            ".env",
            ".env.local",
            "server.pem",
            "credentials.json",
            "credentials.py",
            "config.yaml",
            "settings.py",
        ],
    )
    assert {".env", ".env.local", "server.pem", "credentials.json", "config.yaml"} <= set(ignored)
    assert not {"credentials.py", "settings.py"} & set(ignored)


def test_base_version_compatibility(tmp_path):
    from zipfile import ZipFile

    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "mobilerun_agents-0.1.0.dist-info/METADATA",
            'Metadata-Version: 2.3\nName: mobilerun-agents\nVersion: 0.1.0\nRequires-Dist: mobilerun>=0.7.4; python_version >= "3.11"\n\n',
        )
    mod.verify_base_version(wheel, "0.7.4", [])
    with pytest.raises(ValueError, match="0.7.3"):
        mod.verify_base_version(wheel, "0.7.3", [])


def test_refuses_running_environment_before_build(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["harness", "--base-version", "0.7.4", "--env", str(tmp_path)])
    with pytest.raises(ValueError, match="another Python"):
        mod.main()


def test_version_check_with_packaging_blocked_in_harness(tmp_path):
    from zipfile import ZipFile

    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            "test.dist-info/METADATA",
            'Requires-Dist: mobilerun>=0.7.4; python_version >= "3.11"\n\n',
        )
    code = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "packaging" or fullname.startswith("packaging."):
            raise ModuleNotFoundError("packaging blocked")
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location("harness", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.check_base_version(sys.argv[2], sys.argv[3], "0.7.4", [])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(ROOT / "scripts/prepare_split_env.py"),
            sys.executable,
            str(wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["file", "directory", "top-level"])
def test_staging_rejects_symlinks(tmp_path, kind):
    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        (source / name).write_text("fake")
    for name in ("mobilerun", "skills"):
        (source / name).mkdir()
    target = tmp_path / "external"
    if kind == "directory":
        target.mkdir()
        link = source / "mobilerun" / "linked"
    else:
        target.write_text("fake secret")
        link = source / ("README.md" if kind == "top-level" else "mobilerun/asset.json")
        if link.exists():
            link.unlink()
    link.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="symlink") as error:
        mod.stage_base_source(source, tmp_path / "staged")
    assert str(link) in str(error.value)

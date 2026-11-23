import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_wheel_scope(tmp_path):
    script = ROOT / "scripts/prepare_split_env.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("split_harness", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("mobilerun/__init__.py", "")
    with pytest.raises(ValueError):
        mod.validate_wheel(wheel)
    with ZipFile(wheel, "w") as archive:
        archive.writestr("mobilerun/agents/assets/prompt.md", "prompt")
        archive.writestr("mobilerun_agents-0.1.0.dist-info/METADATA", "")
    assert "mobilerun/agents/assets/prompt.md" in mod.validate_wheel(wheel)


def test_wheel_rejects_traversal_and_other_prefix(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in (
        "mobilerun/agents/../../escape.py",
        "mobilerun/other.py",
        "mobilerun_agents-extra/evil.dist-info/METADATA",
    ):
        wheel = tmp_path / "test.whl"
        with ZipFile(wheel, "w") as archive:
            archive.writestr(name, "")
        with pytest.raises(ValueError):
            mod.validate_wheel(wheel)


def test_actual_wheel_and_sdist_resources(tmp_path):
    import json
    import shutil
    import subprocess
    from zipfile import ZipFile

    spec = importlib.util.spec_from_file_location(
        "split_harness", ROOT / "scripts/prepare_split_env.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel and sdist")
    direct = tmp_path / "direct"
    sdists = tmp_path / "sdists"
    rebuilt = tmp_path / "rebuilt"
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(direct), str(ROOT)], check=True)
    subprocess.run([uv, "build", "--sdist", "--out-dir", str(sdists), str(ROOT)], check=True)
    sdist = next(sdists.glob("*.tar.gz"))
    import tarfile

    with tarfile.open(sdist) as archive:
        assert any(name.endswith("/CONTRIBUTING.md") for name in archive.getnames())
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(rebuilt), str(sdist)], check=True)
    manifest = json.loads((ROOT / "scripts/source_manifest.json").read_text())
    mapping = json.loads((ROOT / "scripts/module_map.json").read_text())
    agent_root = ROOT / "src/mobilerun/agents"
    expected = {"mobilerun/agents/README.md"}
    for canonical in mapping.values():
        rel = canonical.removeprefix("mobilerun.agents").strip(".").replace(".", "/")
        for candidate in (agent_root / (rel + ".py"), agent_root / rel / "__init__.py"):
            if candidate.is_file():
                expected.add("mobilerun/agents/" + candidate.relative_to(agent_root).as_posix())
    for rel in manifest:
        if rel.startswith("assets/") and (agent_root / rel).is_file():
            expected.add("mobilerun/agents/" + rel)
    for directory in (direct, rebuilt):
        wheel = next(directory.glob("*.whl"))
        names = set(mod.validate_wheel(wheel))
        assert expected <= names
        assert "mobilerun/__init__.py" not in names
        with ZipFile(wheel) as archive:
            for name in expected:
                assert archive.read(name) == (ROOT / "src" / name).read_bytes()

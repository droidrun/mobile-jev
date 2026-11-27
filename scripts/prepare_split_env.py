"""Build and install the base and add-on as isolated, non-editable wheels."""

import argparse
import email
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".build"


def run(*args, cwd=ROOT):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def capture(*args, cwd=ROOT):
    return subprocess.check_output([str(x) for x in args], cwd=cwd, text=True).strip()


def validate_wheel(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    for name in names:
        first = name.split("/", 1)[0]
        if (
            ".." in name.split("/")
            or name.startswith("/")
            or not (
                name.startswith("mobilerun_agents/")
                or re.fullmatch(r"mobilerun_agents-[^/]+\.dist-info", first)
                and "/" in name
            )
        ):
            raise ValueError(f"wheel contains unexpected path: {name}")
    if not any(name.startswith("mobilerun_agents/") for name in names):
        raise ValueError("wheel is missing agents files")
    return names


def staging_ignore(directory, names):
    patterns = (
        ".git",
        "__pycache__",
        "*.pyc",
        ".env*",
        "*.pem",
        "*.key",
        "credentials.json",
        "auth-profiles.json",
        "config.yaml",
        "config.yml",
        "config.json",
    )
    return {name for name in names if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)}


def stage_base_source(source_dir, staged):
    names = ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE", "mobilerun", "skills")
    # Validate all entries before copying; rglob does not follow directory symlinks.
    for name in names:
        src = source_dir / name
        if src.is_symlink():
            raise ValueError(f"refusing source symlink: {src}")
        if src.is_dir():
            for path in src.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"refusing source symlink: {path}")
    staged.mkdir()
    for name in names:
        src = source_dir / name
        if src.is_dir():
            shutil.copytree(src, staged / name, ignore=staging_ignore, symlinks=False)
        else:
            shutil.copy2(src, staged / name)


def stage_skills_index(checkout, staged):
    """Give the base build hook a local tracked-file index, never a link to the source Git repo."""
    if not (checkout / ".git").exists():
        raise ValueError("skills Git metadata missing: supply the matching --skills-checkout")
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=checkout)
    run("git", "init", "-q", staged)
    subprocess.run(
        ["git", "update-index", "--info-only", "-z", "--index-info"],
        input=entries,
        cwd=staged,
        check=True,
    )


def wheel_dependencies(wheel, extras):
    from packaging.requirements import Requirement

    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must have exactly one METADATA")
        metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
    dependencies = []
    for raw in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(raw)
        if requirement.name.lower().replace("_", "-") == "mobilerun":
            continue
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in [""] + extras
        ):
            continue
        dependencies.append(raw.split(";", 1)[0].strip())
    return dependencies


def verify_base_version(wheel, version, extras):
    from packaging.requirements import Requirement
    from packaging.version import Version

    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must have exactly one METADATA")
        metadata = email.message_from_bytes(archive.read(names[0]))
    for raw in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(raw)
        if requirement.name.lower().replace("_", "-") != "mobilerun":
            continue
        if requirement.marker and not any(
            requirement.marker.evaluate({"extra": extra}) for extra in [""] + extras
        ):
            continue
        if Version(version) not in requirement.specifier:
            raise ValueError(f"installed mobilerun {version} does not satisfy {requirement}")


def check_base_version(python, wheel, version, extras):
    code = (
        "import importlib.util,json,sys; "
        "s=importlib.util.spec_from_file_location('harness',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.verify_base_version(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]))"
    )
    run(python, "-c", code, __file__, wheel, version, json.dumps(extras))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-source", type=Path)
    source.add_argument("--base-wheel", type=Path)
    source.add_argument("--base-version")
    parser.add_argument(
        "--skills-checkout",
        type=Path,
        help="Skills Git checkout for a Git-free base source snapshot (index only)",
    )
    parser.add_argument("--env", required=True, type=Path)
    parser.add_argument("--extras", default="")
    parser.add_argument("--without-extras", action="store_true")
    parser.add_argument("--addon-wheel", type=Path)
    args = parser.parse_args()
    env = args.env.resolve()
    if env == Path(sys.prefix).resolve():
        raise ValueError(
            "target is the running Python environment; run with another Python, e.g. python3 scripts/prepare_split_env.py ..."
        )
    BUILD.mkdir(exist_ok=True)
    empty = BUILD / "empty-cwd"
    empty.mkdir(exist_ok=True)
    base_wheel = args.base_wheel.resolve() if args.base_wheel else None
    if args.base_source:
        source_dir = args.base_source.resolve()
        staged = BUILD / "base-src"
        if staged.exists():
            shutil.rmtree(staged)
        stage_base_source(source_dir, staged)
        stage_skills_index(args.skills_checkout or source_dir / "skills", staged / "skills")
        out = BUILD / "base-wheel"
        out.mkdir(exist_ok=True)
        run("uv", "build", "--wheel", "--out-dir", out, staged)
        wheels = sorted(out.glob("*.whl"), key=lambda p: p.stat().st_mtime_ns)
        base_wheel = wheels[-1]
    if args.base_version:
        out = BUILD / "published-base-wheel"
        out.mkdir(exist_ok=True)
        run(
            "uv",
            "tool",
            "run",
            "--from",
            "pip",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            out,
            "mobilerun==" + args.base_version,
        )
        wheels = sorted(out.glob("mobilerun-*.whl"), key=lambda p: p.stat().st_mtime_ns)
        if not wheels:
            raise ValueError("published base wheel not downloaded")
        base_wheel = wheels[-1]
    addon = args.addon_wheel.resolve() if args.addon_wheel else None
    if not addon:
        out = BUILD / "addon-wheel"
        out.mkdir(exist_ok=True)
        run("uv", "build", "--wheel", "--out-dir", out, ROOT)
        addon = sorted(out.glob("*.whl"), key=lambda p: p.stat().st_mtime_ns)[-1]
    validate_wheel(addon)
    if env == ROOT or env == Path(env.anchor) or not env.name or env in ROOT.parents:
        raise ValueError("unsafe environment path")
    if env.exists():
        if not (env / "pyvenv.cfg").is_file():
            raise ValueError("refusing to remove a non-venv directory")
        shutil.rmtree(env)
    run("uv", "venv", "--python", getattr(sys, "_base_executable", sys.executable), env)
    python = env / "bin" / "python"
    run("uv", "pip", "install", "--python", python, "packaging")
    base_req = str(base_wheel) if base_wheel else "mobilerun==" + args.base_version
    run("uv", "pip", "install", "--python", python, base_req)
    version = capture(
        python, "-c", "import importlib.metadata as m; print(m.version('mobilerun'))", cwd=empty
    )
    purelib = Path(
        capture(python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))", cwd=empty)
    )
    extras = [e for e in args.extras.split(",") if e]
    if args.without_extras:
        extras = [e for e in extras if e != "langfuse"]
    dependency_code = (
        "import importlib.util,json,sys; "
        "s=importlib.util.spec_from_file_location('harness',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "print(json.dumps(m.wheel_dependencies(sys.argv[2],json.loads(sys.argv[3]))))"
    )
    check_base_version(python, addon, version, extras)
    dependencies = json.loads(
        capture(python, "-c", dependency_code, __file__, addon, json.dumps(extras), cwd=empty)
    )
    if dependencies:
        run("uv", "pip", "install", "--python", python, *dependencies)
    run("uv", "pip", "install", "--python", python, "--no-deps", addon)
    assert (
        capture(
            python, "-c", "import importlib.metadata as m; print(m.version('mobilerun'))", cwd=empty
        )
        == version
    )
    location = capture(
        python,
        "-c",
        "import mobilerun_agents; print(mobilerun_agents.__file__)",
        cwd=empty,
    )
    report = {
        "base_version": version,
        "base_source": str(args.base_source.resolve()) if args.base_source else None,
        "base_wheel": str(base_wheel) if base_wheel else base_req,
        "base_sha256": digest(base_wheel) if base_wheel else None,
        "addon_wheel": str(addon),
        "addon_sha256": digest(addon),
        "purelib": str(purelib),
        "agents_location": location,
    }
    (BUILD / "split-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

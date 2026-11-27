import subprocess
import sys
import sysconfig
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture(scope="session")
def agent_wheel_site(tmp_path_factory) -> Path:
    """Check the installed wheel in CI, or a fresh wheel without changing an editable venv."""
    import mobilerun_agents

    installed = Path(sysconfig.get_paths()["purelib"]).resolve()
    if Path(mobilerun_agents.__file__).resolve().is_relative_to(installed):
        return installed

    root = Path(__file__).resolve().parents[2]
    artifacts = tmp_path_factory.mktemp("agent-wheel")
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(artifacts)],
        cwd=root,
        check=True,
    )
    [wheel] = artifacts.glob("mobilerun_agents-*.whl")
    site = artifacts / "site-packages"
    with ZipFile(wheel) as archive:
        archive.extractall(site)
    return site

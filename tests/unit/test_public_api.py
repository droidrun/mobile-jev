import tomllib
from pathlib import Path

import pytest

import mobilerun
from mobilerun.agents.workflows.droid import MobileAgentState

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_project(path: Path) -> dict:
    with path.open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def test_agent_state_is_canonical_in_agents_and_deprecated_at_root():
    from mobilerun import agents

    assert agents.MobileAgentState is MobileAgentState
    assert "MobileAgentState" in agents.__all__
    assert "MobileAgentState" not in mobilerun.__all__
    with pytest.warns(DeprecationWarning, match="mobilerun.MobileAgentState"):
        assert mobilerun.MobileAgentState is agents.MobileAgentState


@pytest.mark.parametrize(
    "name",
    [
        "MobileAgent",
        "MobileAgentState",
        "DroidAgent",
        "DroidAgentState",
        "load_llm",
        "ResultEvent",
        "MacroPlayer",
    ],
)
def test_base_resolves_legacy_apis_only_with_deprecation(name):
    assert name not in mobilerun.__all__
    with pytest.warns(DeprecationWarning, match=f"mobilerun.{name}"):
        assert getattr(mobilerun, name) is not None


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("mobilerun.config.user_config", "DroidConfig"),
        ("mobilerun.agents.config.user_config", "DroidConfig"),
        ("mobilerun.agents.workflows.droid", "DroidAgent"),
        ("mobilerun.agents.workflows.droid", "DroidAgentState"),
        ("mobilerun.agents.workflows.droid.state", "DroidAgentState"),
        ("mobilerun.agents.workflows.droid.droid_agent", "DroidAgent"),
        ("mobilerun.agents.telemetry", "DroidAgentInitEvent"),
        ("mobilerun.agents.telemetry.events", "DroidAgentFinalizeEvent"),
        ("mobilerun.agents.mcp.adapter", "mcp_to_droidrun_tools"),
    ],
)
def test_removed_legacy_names(module, name):
    from importlib import import_module

    assert not hasattr(import_module(module), name)


def test_library_import_does_not_install_logging_handler():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun, logging; assert not logging.getLogger('mobilerun').handlers",
        ],
        check=True,
    )


def test_legacy_root_access_loads_agents_only_on_demand():
    import subprocess
    import sys

    script = """
import sys
import warnings
import mobilerun
assert not any(name.startswith("mobilerun.agents") for name in sys.modules)
assert "MobileAgentState" not in mobilerun.__all__
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    state = mobilerun.MobileAgentState
assert any(issubclass(item.category, DeprecationWarning) for item in caught)
from mobilerun.agents import MobileAgentState
assert state is MobileAgentState
assert "MobileAgentState" not in vars(mobilerun)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

import subprocess
import sys


def test_configuration_does_not_import_workflow_loop():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from mobilerun.agents import MobileConfig
from mobilerun.agents.llm.providers import list_provider_families
assert MobileConfig and list_provider_families
assert 'mobilerun.agents.workflows.droid.droid_agent' not in sys.modules
assert 'mobilerun.agents.workflows.fast_agent.fast_agent' not in sys.modules
""",
        ],
        check=True,
    )

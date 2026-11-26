def test_library_import_does_not_install_logging_handler():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun_agents, logging; assert not logging.getLogger('mobilerun').handlers",
        ],
        check=True,
    )


def test_base_import_does_not_load_agents():
    import subprocess
    import sys

    script = """
import sys
import mobilerun
assert not any(name.startswith("mobilerun_agents") for name in sys.modules)
assert not hasattr(mobilerun, "MobileAgentState")
assert not any(name.startswith("mobilerun_agents") for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

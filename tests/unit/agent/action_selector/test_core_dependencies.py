import importlib.metadata
import subprocess
import sys

from packaging.requirements import Requirement


def test_basic_providers_are_unconditional_dependencies():
    requirements = [Requirement(value) for value in importlib.metadata.requires("mobilerun-agents")]
    for package in ("typesafe-sdk", "anthropic", "llama-index-llms-anthropic", "openai"):
        requirement = next(item for item in requirements if item.name == package)
        assert requirement.marker is None
    extras = importlib.metadata.metadata("mobilerun-agents").get_all("Provides-Extra")
    assert set(extras) == {"dev", "langfuse"}


def test_core_provider_imports_in_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anthropic, typesafe_sdk; from llama_index.llms.anthropic import Anthropic",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

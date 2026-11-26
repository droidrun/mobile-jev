from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from mobilerun.cli.devices.base.options import DeviceTarget
from mobilerun.models.results import ExecutionResult

from mobilerun_agents import AgentOperator, PlanningOperator
from mobilerun_agents.cli.agent.run import run
from mobilerun_agents.config.loader import ConfigLoader
from tests.unit.device_fixtures import MemoryDevice


@pytest.mark.parametrize("args", [[], ["phone-id", "--cloud"], ["phone-id", "--cloud", "--plan"]])
def test_no_model_precedes_device_discovery_consent_and_open(monkeypatch, args):
    forbidden = Mock(side_effect=AssertionError("device accessed before model resolution"))
    monkeypatch.setattr("mobilerun.cli.devices.base.options.discover_devices", forbidden)
    monkeypatch.setattr("mobilerun_agents.cli.agent.run.confirm_action", forbidden)
    monkeypatch.setattr(DeviceTarget, "open", forbidden)
    result = CliRunner().invoke(run, ["goal", *args])
    assert (
        result.exit_code == 1 and "No model configured. Run: mobilerun agent setup" in result.output
    )
    forbidden.assert_not_called()


@pytest.mark.parametrize("model", [None, "gpt-5-mini"])
def test_run_forwards_overrides(monkeypatch, tmp_path, model):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    phone = MemoryDevice()
    calls = []

    @asynccontextmanager
    async def opened(target):
        yield phone

    async def execute(operator, device, goal, **options):
        calls.append((operator, ConfigLoader.load(), device, goal, options))
        return ExecutionResult(success=True, answer="finished", device="memory")

    monkeypatch.setattr(DeviceTarget, "open", opened)
    monkeypatch.setattr(AgentOperator, "execute", execute)
    monkeypatch.setattr(PlanningOperator, "execute", execute)
    path = tmp_path / "agents.yaml"
    path.write_text(
        "agent: {selector: private, delegation: {delegate: helper}}\n"
        "llm_profiles:\n"
        "  private: {provider: openai_like, model: local, base_url: 'https://private.invalid', kwargs: {private: true}}\n"
        "  hosted: {provider: openai, model: gpt-4o, temperature: 0.7}\n"
    )
    out = CliRunner().invoke(
        run,
        [
            "count things",
            "emulator-5554",
            "--config",
            str(path),
            "-p",
            "openai",
            *(["-m", model] if model else []),
            "--steps",
            *(["2", "--plan"] if model else ["2"]),
            "--no-delegate",
        ],
    )
    assert out.exit_code == 0, out.exception
    op, config, device, goal, options = calls[0]
    assert device is phone and goal == "count things"
    assert type(op) is (PlanningOperator if model else AgentOperator) and options["max_steps"] == 2
    profile = config.llm_profiles[config.agent.selector]
    assert (profile.provider, profile.model, config.agent.delegation) == (
        "openai",
        model or "gpt-4o",
        None,
    )
    assert config.agent.selector == "hosted"
    assert (profile.base_url, profile.kwargs, profile.temperature) == (None, {}, 0.7)
    assert config.llm_profiles["private"].model == "local"
    other = CliRunner().invoke(
        run, ["goal", "emulator-5554", "--config", str(path), "-p", "ollama"]
    )
    assert other.exit_code == 2 and "not configured" in other.output
    assert len(calls) == 1


def test_explicit_provider_requires_a_saved_profile(monkeypatch):
    def unexpected_open(target):
        raise AssertionError("Missing provider must fail before opening a device")

    monkeypatch.setattr(DeviceTarget, "open", unexpected_open)
    out = CliRunner().invoke(run, ["goal", "emulator-5554", "--provider", "gemini"])
    assert out.exit_code == 2 and "not configured" in out.output

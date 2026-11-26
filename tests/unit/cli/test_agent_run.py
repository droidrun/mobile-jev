from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner
from llama_index.core.llms import MockLLM
from mobilerun.cli.devices.base.options import DeviceTarget
from mobilerun.models.results import ExecutionResult

from mobilerun_agents import AgentOperator, PlanningOperator
from mobilerun_agents.agent.action_selector import loader
from mobilerun_agents.agent.action_selector.decision_model.base import DecisionSelector
from mobilerun_agents.agent.action_selector.delegating import DelegatingSelector
from mobilerun_agents.agent.action_selector.llm.llama_index import LlamaIndexLLMSelector
from mobilerun_agents.cli.agent.run import run
from mobilerun_agents.config.loader import ConfigLoader
from tests.unit.device_fixtures import MemoryDevice


def test_keyboard_interrupt_exits_with_130_without_traceback(monkeypatch):
    class InterruptedRunner:
        def __enter__(self):
            raise KeyboardInterrupt

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("mobilerun_agents.cli.agent.run.asyncio.Runner", InterruptedRunner)
    result = CliRunner().invoke(run, ["do something", "serial"])
    assert result.exit_code == 130
    assert "Traceback" not in result.output


@pytest.mark.parametrize("args", [[], ["phone-id", "--cloud"], ["phone-id", "--setup", "missing"]])
def test_setup_error_precedes_device_discovery_consent_and_open(monkeypatch, args):
    forbidden = Mock(side_effect=AssertionError("device accessed before model resolution"))
    monkeypatch.setattr("mobilerun.cli.devices.base.options.discover_devices", forbidden)
    monkeypatch.setattr(DeviceTarget, "open", forbidden)
    result = CliRunner().invoke(run, ["goal", *args])
    expected = "Unknown setup 'missing'" if "missing" in args else "No model configured."
    assert result.exit_code == 1 and expected in result.output
    forbidden.assert_not_called()


@pytest.mark.parametrize("name", ["fast", "deep", "jev-only"])
def test_run_selects_setup_and_overrides_vision_without_writing(monkeypatch, name):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    path = ConfigLoader.get_user_config_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        "profiles:\n"
        "  luna: {provider: ollama, model: local, vision: off}\n"
        "  sol: {provider: ollama, model: planner}\n"
        "  decision: {provider: typesafe, vision: on}\n"
        "setups:\n"
        "  fast: {selector: luna}\n"
        "  deep: {planner: sol, executor: decision, delegate: luna}\n"
        "  jev-only: {selector: decision}\n"
        "default_setup: fast\n"
    )
    saved = path.read_bytes()
    phone, calls = MemoryDevice(), []
    build = AsyncMock(return_value=MockLLM())
    monkeypatch.setattr(loader, "build_llm", build)

    @asynccontextmanager
    async def opened(target):
        yield phone

    async def execute(operator, device, goal, **options):
        calls.append((operator, device, goal, options))
        return ExecutionResult(success=True, answer="finished", device="memory")

    monkeypatch.setattr(DeviceTarget, "open", opened)
    monkeypatch.setattr(AgentOperator, "execute", execute)
    monkeypatch.setattr(PlanningOperator, "execute", execute)
    result = CliRunner().invoke(
        run, ["count things", "emulator-5554", "--setup", name, "--vision", "only", "--steps", "2"]
    )
    assert result.exit_code == 0, result.output
    [(op, device, goal, options)] = calls
    assert device is phone and goal == "count things" and options["max_steps"] == 2
    if name == "deep":
        assert type(op) is PlanningOperator and type(op._executor) is DelegatingSelector
        assert op._planner.llm is build.return_value
        assert op._executor.primary.vision == op._executor.delegate.vision == "only"
        assert [call.args[1] for call in build.await_args_list] == ["planner", "local"]
    else:
        expected = LlamaIndexLLMSelector if name == "fast" else DecisionSelector
        assert type(op) is AgentOperator and type(op._selector) is expected
        assert op._selector.vision == "only"
    assert path.read_bytes() == saved

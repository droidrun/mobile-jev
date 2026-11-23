from click.testing import CliRunner

from tests.unit.cli_macro_fixtures import fake_device_factory
from tests.unit.macro_operator_fixtures import write_macro


def test_agent_replay_finishes_on_mismatch(monkeypatch, tmp_path):
    from mobilerun.cli.app import cli
    from mobilerun.models.results import ExecutionResult

    from mobilerun.agents.operators.agent import AgentOperator

    device, _ = fake_device_factory(monkeypatch)
    calls = []

    async def execute(self, supplied_device, instruction, **kwargs):
        calls.append((supplied_device, instruction, self.config))
        return ExecutionResult(success=True, device="fake")

    monkeypatch.setattr(AgentOperator, "execute", execute)
    result = CliRunner().invoke(
        cli,
        [
            "agent",
            "replay",
            str(write_macro(tmp_path, True)),
            "serial",
            "--state-timeout",
            "0",
            "--provider",
            "OpenAI",
            "--model",
            "fake-model",
        ],
    )
    assert result.exit_code == 0, result.output
    assert device.taps == []
    assert len(calls) == 1
    assert calls[0][0] is device
    assert "finish task" in calls[0][1]
    assert '"action_type": "tap"' in calls[0][1]
    assert calls[0][2].llm_profiles["fast_agent"].model == "fake-model"

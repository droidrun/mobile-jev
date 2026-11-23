import asyncio

from mobilerun.models.results import ExecutionResult

from tests.unit.macro_operator_fixtures import FakeDevice, write_macro


def test_agent_macro_resumes_on_same_device_with_remaining_steps(tmp_path):
    from mobilerun.agents import AgentMacroOperator

    calls = []

    class FakeAgent:
        async def execute(self, device, instruction, **kwargs):
            calls.append((device, instruction))
            return ExecutionResult(success=True, device=type(device).__name__)

    device = FakeDevice()
    result = asyncio.run(
        AgentMacroOperator(agent=FakeAgent(), state_timeout=0).execute(
            device, str(write_macro(tmp_path, True))
        )
    )
    assert result.success
    assert device.taps == []
    assert len(calls) == 1
    assert calls[0][0] is device
    assert "finish task" in calls[0][1]
    assert '"action_type": "tap"' in calls[0][1]


def test_agent_macro_does_not_invoke_agent_when_replay_matches(tmp_path):
    from mobilerun.agents import AgentMacroOperator

    class UnavailableAgent:
        async def execute(self, *args, **kwargs):
            raise AssertionError("Agent must not run for deterministic replay")

    device = FakeDevice()
    result = asyncio.run(
        AgentMacroOperator(agent=UnavailableAgent()).execute(device, str(write_macro(tmp_path)))
    )
    assert result.success
    assert device.taps == [(10, 20)]


def test_matching_replay_extracts_typed_output_from_current_screen(tmp_path):
    from mobilerun.agents import AgentMacroOperator

    class Output:
        pass

    class FakeAgent:
        async def execute(self, device, instruction, **kwargs):
            assert device.taps == [(10, 20)]
            assert "finish task" in instruction
            assert "current" in instruction.lower()
            assert kwargs == {"output": Output, "max_steps": 2}
            return ExecutionResult(success=True, answer="extracted", device="FakeDevice")

    device = FakeDevice()
    result = asyncio.run(
        AgentMacroOperator(agent=FakeAgent()).execute(
            device, str(write_macro(tmp_path)), output=Output, max_steps=3
        )
    )
    assert result.answer == "extracted"
    assert device.taps == [(10, 20)]

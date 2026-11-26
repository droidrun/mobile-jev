import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from mobilerun.macro.errors import MacroError, MacroMismatch
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun_agents.macro_agent import MacroAgentOperator, operator


class Output(BaseModel):
    title: str


def test_base_player_stops_before_guarded_action(tmp_path):
    class Device:
        state = None
        tap = AsyncMock()

        async def _get_ui_tree(self):
            return {"phone_state": {"package": "unexpected"}}

    actions = [
        {"action_type": "tap", "x": 1, "y": 2},
        {
            "action_type": "tap",
            "x": 3,
            "y": 4,
            "pre_state": {"phone_state": {"package": "inbox"}, "nodes": [{"text": "Send"}]},
        },
        {"action_type": "tap", "x": 5, "y": 6},
    ]
    (tmp_path / "macro.json").write_text(
        json.dumps({"macro_schema_version": "2.0", "actions": actions})
    )
    agent = Mock(
        execute=AsyncMock(return_value=ExecutionResult(success=True, device="fake", steps=2))
    )
    device = Device()
    result = asyncio.run(
        MacroAgentOperator(tmp_path, agent=agent, state_timeout=0, delay_between_actions=0).execute(
            device, "Send email"
        )
    )
    assert result.success and result.steps == 3
    device.tap.assert_awaited_once_with(1, 2)
    assert json.dumps(actions[1:]) in agent.execute.await_args.args[1]


@pytest.fixture
def replay(monkeypatch):
    player = Mock(executed_steps=2, replay_macro=AsyncMock(return_value=True))
    player.load_macro_from_file.return_value = {"description": "open inbox"}
    monkeypatch.setattr(operator, "MacroPlayer", Mock(return_value=player))
    agent = Mock(execute=AsyncMock())
    return object(), player, agent


def test_matching_run_folder_never_calls_agent(replay, tmp_path):
    device, player, agent = replay
    result = asyncio.run(MacroAgentOperator(agent=agent).execute(device, str(tmp_path)))
    player.load_macro_from_file.assert_called_once_with(str(tmp_path))
    assert result.success and result.steps == 2
    agent.execute.assert_not_called()


@pytest.mark.parametrize(
    ("instruction", "description", "goal"),
    [
        ("my goal", "recorded", "my goal"),
        (None, "recorded", "recorded"),
        (None, None, "Finish the recorded task"),
    ],
)
def test_divergence_hands_off_once_and_preserves_result(replay, instruction, description, goal):
    device, player, agent = replay
    player.load_macro_from_file.return_value = {"description": description}
    player.replay_macro.side_effect = MacroMismatch(2, [{"action_type": "tap"}], {"screen": "home"})
    recovered = ExecutionResult[Output](
        success=False, device="fake", answer="stopped", steps=1, task_id="id"
    )
    agent.execute.return_value = recovered
    op = MacroAgentOperator("macro.json" if instruction else None, agent=agent)
    actual = asyncio.run(
        op.execute(device, instruction or "macro.json", output=Output, max_steps=5)
    )
    assert actual == recovered.model_copy(update={"steps": 3})
    agent.execute.assert_awaited_once()
    target, prompt = agent.execute.await_args.args
    assert target is device
    assert prompt.startswith(goal) and '"action_type": "tap"' in prompt
    assert '"screen": "home"' in prompt and "step 3" in prompt
    assert agent.execute.await_args.kwargs["max_steps"] == 3
    assert agent.execute.await_args.kwargs["output"] is Output
    player.replay_macro.assert_awaited_once()


def test_matching_macro_with_output_extracts_without_repeating(replay):
    device, _, agent = replay
    agent.execute.return_value = ExecutionResult[Output](
        success=True, device="fake", output=Output(title="Inbox"), steps=1
    )
    result = asyncio.run(
        MacroAgentOperator(agent=agent).execute(device, "macro.json", output=Output)
    )
    assert result.steps == 3 and result.output.title == "Inbox"
    assert "without repeating" in agent.execute.await_args.args[1]


def test_action_error_does_not_handoff(replay):
    device, player, agent = replay
    player.replay_macro.side_effect = MacroError("missing extension")
    with pytest.raises(MacroError, match="missing extension"):
        asyncio.run(MacroAgentOperator(agent=agent).execute(device, "macro.json"))
    agent.execute.assert_not_called()

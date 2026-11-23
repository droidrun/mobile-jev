"""Completion findings survive both coordination paths without status inference."""

import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.agents.workflows.droid.droid_agent import MobileAgent
from mobilerun.agents.workflows.droid.events import FastAgentResultEvent, ManagerPlanEvent
from mobilerun.agents.workflows.droid.state import MobileAgentState


@pytest.mark.parametrize("message", ["Number 7 is skipped", "done", ""])
def test_fast_completion_carries_answer_separately(message):
    state = MobileAgentState()
    asyncio.run(state.complete(True, reason="finished", message=message))
    assert state.completion_reason == "finished"
    assert state.answer == message
    agent = object.__new__(MobileAgent)
    event = asyncio.run(
        agent.handle_fast_agent_result(
            None,
            FastAgentResultEvent(
                success=True,
                reason=state.completion_reason,
                answer=state.answer or None,
                instruction="find",
            ),
        )
    )
    assert event.reason == "finished"
    assert event.answer == (message or None)


def test_manager_completion_has_typed_answer_and_status():
    agent = object.__new__(MobileAgent)
    agent.shared_state = SimpleNamespace(pending_user_messages=[], progress_summary="")
    event = asyncio.run(
        agent.handle_manager_plan(
            None,
            ManagerPlanEvent(
                plan="",
                current_subgoal="",
                thought="",
                success=True,
                answer="Number 7 is skipped",
            ),
        )
    )
    assert event.answer == "Number 7 is skipped"
    assert event.reason == "Task completed successfully."


def test_failure_message_remains_failure_reason():
    state = MobileAgentState()
    asyncio.run(state.complete(False, message="No counter found"))
    assert state.completion_reason == "No counter found"


def test_failure_keeps_explicit_reason_separate_from_message():
    state = MobileAgentState()
    asyncio.run(state.complete(False, reason="Counter unavailable", message="Number 7 is skipped"))
    assert state.completion_reason == "Counter unavailable"
    assert state.answer == "Number 7 is skipped"


def test_legacy_status_does_not_invent_answer():
    state = MobileAgentState()
    asyncio.run(state.complete(True, reason="done"))
    assert state.answer == ""

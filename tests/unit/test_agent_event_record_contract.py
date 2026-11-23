from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

from llama_index.core.workflow import Event

from mobilerun.agents.workflows.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agents.workflows.droid.droid_agent import MobileAgent
from mobilerun.agents.workflows.fast_agent.events import FastAgentOutputEvent


def test_agent_events_record_through_their_own_contract():
    trajectory = SimpleNamespace(
        screenshot_queue=deque(), screenshot_count=0, ui_states=[], events=[]
    )
    writer = SimpleNamespace(write=Mock())
    screenshot = ScreenshotEvent(screenshot=b"image")
    state = RecordUIStateEvent(ui_state=[{"text": "a"}])
    output = FastAgentOutputEvent(output="ok")
    screenshot.record(trajectory, writer, None)
    state.record(trajectory, writer, None)
    output.record(trajectory, writer, 3)
    assert list(trajectory.screenshot_queue) == [b"image"]
    assert trajectory.screenshot_count == 1
    assert trajectory.ui_states == [[{"text": "a"}]]
    assert trajectory.events == [output]
    writer.write.assert_called_once_with(trajectory, stage="fast_agent_step_3")


def test_external_workflow_events_remain_recorded():
    agent = object.__new__(MobileAgent)
    agent.trajectory = SimpleNamespace(events=[])
    agent.trajectory_writer = None
    events = [Event(), Event(record="third-party data")]
    ctx = SimpleNamespace(write_event_to_stream=Mock())
    for event in events:
        agent.handle_stream_event(event, ctx)
    assert agent.trajectory.events == events
    assert ctx.write_event_to_stream.call_count == 2

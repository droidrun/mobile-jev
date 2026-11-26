import pytest
from llama_index.core.base.llms.types import ImageBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from mobilerun.core.state.ui_state import UIState
from pydantic import PrivateAttr
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.action_selector.llm.models import Capabilities
from mobilerun_agents.agent.planner import Finish, LlamaIndexPlanner, Plan
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import PlanningState, SubgoalResult


class PlanningProbe(Workflow):
    @step
    async def start(self, ctx: Context[PlanningState], ev: StartEvent) -> StopEvent:
        return StopEvent()


class Replies(MockLLM):
    _requests = PrivateAttr(default_factory=list)

    async def achat(self, messages, tools, tool_required, **kwargs):
        assert tool_required
        self._requests.append((messages, tools))
        calls = [
            ("set_plan", {"subgoals": [" "]}),
            ("set_plan", {"subgoals": [" Read battery "]}),
            ("complete", {"success": True, "message": "Battery at 100%"}),
        ]
        name, args = calls[len(self._requests) - 1]
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                blocks=[ToolCallBlock(tool_name=name, tool_kwargs=args, tool_call_id="reply")],
            )
        )

    async def astream_chat(self, messages, **kwargs):
        async def chunks():
            yield await self.achat(messages, **kwargs)

        return chunks()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_llama_index_requires_a_valid_planner_tool_call():
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(PlanningState(goal="Check the battery"))
    llm = Replies()
    planner = LlamaIndexPlanner(llm)

    assert await planner.plan(ctx) == Plan(("Read battery",))
    assert await planner.plan(ctx) == Finish(True, "Battery at 100%")
    messages, tools = llm._requests[1]
    assert [tool.metadata.get_name() for tool in tools] == ["set_plan", "complete"]
    assert "Goal: Check the battery" in messages[0].content
    assert "previous response was invalid" in messages[-1].content


@pytest.mark.anyio
async def test_llm_gets_observation_note_and_path_backed_screenshot(tmp_path, monkeypatch):
    path = tmp_path / "screen.png"
    path.write_bytes(b"fake screenshot payload")
    monkeypatch.setattr(
        "mobilerun_agents.agent.planner.llama_index.adapter_capabilities",
        lambda llm: Capabilities(vision=True),
    )
    ctx = Context[PlanningState](PlanningProbe())
    await ctx.store.set_state(
        PlanningState(
            goal="Check the battery",
            current=StepSnapshot(
                ui=UIState([], "Battery 100%", "", {}, 100, 100), screenshot=ImageBlock(path=path)
            ),
            results=[
                SubgoalResult(subgoal="Read battery", success=True, message="", reason="complete")
            ],
        )
    )
    llm = Replies()
    await LlamaIndexPlanner(llm).bind(None, None, True).plan(ctx)
    messages, _ = llm._requests[0]
    assert "The last subgoal ended without a description" in messages[0].content
    assert "Battery 100%" in messages[0].content
    images = [block for block in messages[0].blocks if isinstance(block, ImageBlock)]
    assert [image.path for image in images] == [path]

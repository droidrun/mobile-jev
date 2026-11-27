import re
from io import BytesIO

import pytest
from llama_index.core.base.llms.types import TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, ImageBlock
from llama_index.core.memory.memory import DEFAULT_MEMORY_BLOCKS_TEMPLATE
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from PIL import Image

from mobilerun_agents.agent.action_selector import DecisionSelector
from mobilerun_agents.agent.action_selector.base import NativeToolAPI
from mobilerun_agents.agent.action_selector.decision_model.base import Decision
from mobilerun_agents.agent.action_selector.decision_model.images import prepare_image
from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.events import Detail, Observed
from mobilerun_agents.agent.logs import ConsoleLog
from mobilerun_agents.agent.memory.memory import build_executor_memory
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord
from mobilerun_agents.config.user_config import DecisionConfig, MobileConfig
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import fake_tools


@pytest.fixture
def image():
    stream = BytesIO()
    Image.new("RGB", (100, 200)).save(stream, "PNG")
    return stream.getvalue()


@pytest.mark.anyio
async def test_executor_request_renders_default_memory_order():
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            system, memory, step = messages
            assert [message.role for message in messages] == ["system", "user", "user"]
            assert "Mobilerun" in system.content
            assert re.findall(r"^<(\w+)>$", memory.content, re.M) == [
                "memory",
                "setup",
                "goal",
                "last_step",
                "history",
                "app_cards",
            ]
            assert "Inbox (current observation)" not in memory.content
            assert "Inbox (current observation)" in step.content
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=('<function_calls><invoke name="wait"></invoke></function_calls>'),
                )
            )

    phone = {"packageName": "com.google.android.gm", "activityName": "Inbox"}
    turn = Turn(
        goal="Find order",
        tools=fake_tools("wait"),
        ui=UIState([], "Inbox (current observation)", "", phone, 100, 200),
    )
    before = StepSnapshot(ui=UIState([], "Previous inbox", "", phone, 100, 200))
    turn.state.actions = [
        ActionRecord(
            step=step,
            call=ToolSelection(tool_id=str(step), tool_name="wait", tool_kwargs={}),
            ok=True,
            summary="Waited",
            before=before,
            screen_changed=True,
        )
        for step in (1, 2)
    ]
    turn.memory = build_executor_memory(MobileConfig(), "android")
    try:
        assert (await turn.select(Chat("off"))).calls[0].tool_name == "wait"
    finally:
        await turn.memory.aclose()


@pytest.mark.parametrize(
    "vision,has_tree,has_image", [("off", True, False), ("on", True, True), ("only", False, True)]
)
@pytest.mark.anyio
async def test_jev_builds_request_from_observation_registry_and_memory(
    vision, has_tree, has_image, image
):
    class Model:
        max_options = None
        locates_points = True

        async def decide(self, request):
            assert (request.state is not None) is has_tree
            assert request.screenshot == (image if has_image else None)
            assert request.goal == "Use battery"
            assert request.memory == "checked settings"
            assert "retry here" in request.previous_action
            assert "63" in request.previous_action
            assert ("tap_element" in request.registry) is has_tree
            assert ("tap" in request.registry) is has_image
            return Decision(
                option_index=next(
                    i for i, option in enumerate(request.options) if option.action.name == "wait"
                )
            )

    turn = Turn(
        goal="Use battery",
        tools=fake_tools("tap", "tap_element", "wait"),
        screenshot=image,
        elements=[{"index": 1}],
    )
    turn.state.repair = "retry here"
    turn.state.current.device_context = [
        ContextItem(source="battery", kind="status", data={"level": 63})
    ]
    turn.memory.aget.return_value = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="checked settings"),
                ImageBlock(image=image),
            ],
        )
    ]
    result = await turn.select(DecisionSelector(Model(), vision=vision))
    turn.memory.aget.assert_awaited_once_with(state=turn.state)
    assert result.calls[0].tool_name == "wait"


@pytest.mark.parametrize(
    "vision,tool,has_image",
    [("off", "tap_element", False), ("on", "tap_element", True), ("only", "tap", True)],
)
@pytest.mark.anyio
async def test_llm_uses_vision_specific_inputs_and_tools(vision, tool, has_image, image):
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            assert any(isinstance(block, ImageBlock) for block in messages[-1].blocks) is has_image
            assert ("screen" in messages[-1].content) is (vision != "only") or has_image
            assert messages[-2].content == "checked settings"
            assert [message.role for message in messages] == ["system", "user", "user"]
            assert "retry here" in messages[-1].content
            assert "63" in messages[-1].content
            assert any(isinstance(block, ImageBlock) for block in messages[-2].blocks) is has_image
            args = (
                '<parameter name="index">1</parameter>'
                if tool == "tap_element"
                else '<parameter name="x">20</parameter><parameter name="y">40</parameter>'
            )
            text = f'<function_calls><invoke name="{tool}">{args}</invoke></function_calls>'
            return ChatResponse(message=ChatMessage(role="assistant", content=text))

    turn = Turn(
        tools=fake_tools("tap", "tap_element", "wait"), screenshot=image, elements=[{"index": 1}]
    )
    turn.state.repair = "retry here"
    turn.state.current.device_context = [
        ContextItem(source="battery", kind="status", data={"level": 63})
    ]
    turn.memory.aget.return_value = [
        ChatMessage(
            role="user",
            blocks=[
                TextBlock(text="checked settings"),
                ImageBlock(image=image),
            ],
        )
    ]
    result = await turn.select(Chat(vision))
    turn.memory.aget.assert_awaited_once_with(state=turn.state)
    assert result.calls[0].tool_name == tool


@pytest.mark.anyio
@pytest.mark.parametrize("api", ["responses", "chat_completions", "anthropic", "gemini"])
async def test_native_requests_keep_system_text_and_prior_image_in_supported_roles(api, image):
    class Chat(LLMSelector):
        @property
        def supports_native_tool_calls(self) -> bool:
            return True

        @property
        def native_tool_api(self) -> NativeToolAPI:
            return api

        async def _achat(self, messages, **kwargs):
            system, memory, step = messages
            assert [message.role for message in messages] == ["system", "user", "user"]
            assert "previous action" in system.content
            assert all(isinstance(block, TextBlock) for block in system.blocks)
            assert memory.blocks == [previous_image]
            assert step.content and any(isinstance(block, ImageBlock) for block in step.blocks)
            assert kwargs["tool_required"] is (api == "responses")
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(tool_name="wait", tool_call_id="reply", tool_kwargs={}),
                    ],
                )
            )

    previous_image = ImageBlock(image=image)
    turn = Turn(tools=fake_tools("wait"), screenshot=image)
    turn.memory.aget.return_value = [
        ChatMessage(
            role="system",
            blocks=[
                TextBlock(text="previous action"),
                previous_image,
            ],
        )
    ]
    assert (await turn.select(Chat())).calls[0].tool_name == "wait"


@pytest.mark.anyio
async def test_llm_rejects_tool_excluded_by_vision():
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            reply = '<function_calls><invoke name="tap"><parameter name="x">20</parameter><parameter name="y">40</parameter></invoke></function_calls>'
            return ChatResponse(message=ChatMessage(role="assistant", content=reply))

    with pytest.raises(InvalidResponse):
        await Turn(tools=fake_tools("tap")).select(Chat("off"))


def test_typesafe_image_limits_and_coverage(image):
    data, coverage = prepare_image(image, DecisionConfig())
    with Image.open(BytesIO(data)) as padded:
        assert padded.width >= 280
        assert padded.height >= 280
    assert 0 < coverage[0] <= 1 and 0 < coverage[1] <= 1


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["llm", "jev"])
@pytest.mark.parametrize("level", ["DEBUG", "INFO"])
async def test_selector_reports_only_debug_memory(kind, level, caplog, tmp_path):
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            return None

    class Model:
        max_options = None
        locates_points = False

        async def decide(self, request):
            return None

    class DiagnosticMemory:
        memory_blocks = [type("Goal", (), {"name": "goal"})()]
        token_limit = 24000
        estimated = False

        async def aget(self, **kwargs):
            return history

        def _estimate_token_count(self, messages):
            self.estimated = True
            return 1900

    history = DEFAULT_MEMORY_BLOCKS_TEMPLATE.format_messages(
        memory_blocks=[("goal", [TextBlock(text="Find order\nfull block second line")])]
    )
    caplog.set_level(level, logger="mobilerun_agents.agent.logs.console")
    turn = Turn(tools=fake_tools("wait"))
    turn.memory = DiagnosticMemory()
    await turn.select(Chat("off") if kind == "llm" else DecisionSelector(Model(), vision="off"))
    details = [e for e in turn.events if isinstance(e, Detail) and e.section == "memory"]
    if level == "INFO":
        assert not details
        assert not turn.memory.estimated
        return

    (detail,) = details
    assert "Find order\nfull block second line" in detail.text
    console = ConsoleLog(tmp_path)
    console.event(Observed(step=31))
    console.event(detail)
    (compact,) = [message for message in caplog.messages if "memory: " in message]
    assert "1.9k/24k tokens · blocks: goal(2 lines)" in compact
    assert len(compact.splitlines()) <= 6
    assert "goal: Find order" in compact
    assert "full block second line" not in compact
    assert "msgs" not in compact
    saved = (tmp_path / "memory/0031.txt").read_text()
    assert "Find order\nfull block second line" in saved

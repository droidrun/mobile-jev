from io import BytesIO

import pytest
from llama_index.core.base.llms.types import TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, ImageBlock
from llama_index.core.memory.memory import DEFAULT_MEMORY_BLOCKS_TEMPLATE
from PIL import Image

from mobilerun_agents.agent.action_selector import DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.base import Decision
from mobilerun_agents.agent.action_selector.decision_model.images import prepare_image
from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.events import Detail, Observed
from mobilerun_agents.agent.logs import ConsoleLog
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.config.user_config import DecisionConfig
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import fake_tools


@pytest.fixture
def image():
    stream = BytesIO()
    Image.new("RGB", (100, 200)).save(stream, "PNG")
    return stream.getvalue()


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
    turn.memory.aget.return_value = [ChatMessage(role="assistant", content="checked settings")]
    result = await turn.select(DecisionSelector(Model(), vision=vision))
    turn.memory.aget.assert_awaited_once_with(snapshot=turn.state.current, state=turn.state)
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
    turn.memory.aget.return_value = [ChatMessage(role="assistant", content="checked settings")]
    result = await turn.select(Chat(vision))
    turn.memory.aget.assert_awaited_once_with(snapshot=turn.state.current, state=turn.state)
    assert result.calls[0].tool_name == tool


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
        memory_blocks = [type("Notes", (), {"name": "notes"})()]
        token_limit = 24000
        estimated = False

        async def aget(self, **kwargs):
            return history

        def _estimate_token_count(self, messages):
            self.estimated = True
            return 1900

    history = DEFAULT_MEMORY_BLOCKS_TEMPLATE.format_messages(
        memory_blocks=[("notes", [TextBlock(text="fixture secret\nfull block second line")])]
    )
    for step in range(1, 31):
        history.extend(
            [
                ChatMessage(role="user", content="Action taken:"),
                ChatMessage(
                    role="assistant",
                    blocks=[
                        ToolCallBlock(
                            tool_call_id=str(step),
                            tool_name="tap_element",
                            tool_kwargs={"index": 12},
                        )
                    ],
                ),
                ChatMessage(role="tool", content=f"{step} · tap_element [12] ✓\nresult at {step}"),
            ]
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
    assert "fixture secret\nfull block second line" in detail.text
    assert 'tap_element({"index": 12})' in detail.text
    assert "result at 1" in detail.text and "result at 30" in detail.text
    redact = Redactor()
    redact.add("fixture secret")
    console = ConsoleLog(redact, tmp_path)
    console.event(Observed(step=31))
    console.event(detail)
    (compact,) = [message for message in caplog.messages if "memory: " in message]
    assert "91 msgs · 1.9k/24k tokens · blocks: notes(2 lines)" in compact
    assert len(compact.splitlines()) <= 6
    assert "notes: ***" in compact
    assert "full block second line" not in compact
    for step in (28, 29, 30):
        assert f"{step} tap_element[12] ✓" in compact
    assert "27 tap_element" not in compact
    assert "fixture secret" not in caplog.text
    saved = (tmp_path / "memory/0031.txt").read_text()
    assert "***\nfull block second line" in saved
    assert "fixture secret" not in saved

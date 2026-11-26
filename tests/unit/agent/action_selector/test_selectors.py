import asyncio
from io import BytesIO

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from PIL import Image

from mobilerun_agents.agent.action_selector import (
    ActionSelection,
    ActionSelector,
    DecisionSelector,
    DelegatingSelector,
    FallbackSelector,
)
from mobilerun_agents.agent.action_selector.decision_model.base import Decision
from mobilerun_agents.agent.action_selector.decision_model.images import prepare_image
from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.config.user_config import DecisionConfig
from tests.unit.agent.selector_fixtures import selection_run
from tests.unit.device_fixtures import fake_tools


class StaticSelector(ActionSelector):
    def __init__(self, selection):
        self.selection = selection
        self.requests = []

    async def select(self, run):
        self.requests.append(run)
        return self.selection or ActionSelection(abstained=True)


class FakeModel:
    max_options = 2
    screenshot_max_side = 1568
    locates_points = True

    async def decide(self, request):
        self.request = request
        return Decision(option_index=0, point=(1, 1), confidence=0.9)


@pytest.mark.anyio
@pytest.mark.parametrize("package_key", ["package", "package_name"])
async def test_jev_offers_and_opens_installed_app_once_listed(package_key):
    from unittest.mock import AsyncMock

    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import MemoryDevice

    class Phone(MemoryDevice):
        async def start_app(self, package: str, activity: str | None = None) -> str:
            self.launched = package
            return package

    run = selection_run("Open Clock")
    phone = Phone(supported_operations=frozenset({"start_app"}))
    phone._set_state_provider(run.state_provider)
    run.device = phone
    phone.list_apps = AsyncMock(return_value=[{package_key: "example.clock", "label": "Clock"}])
    run.registry = build_registry(run, [], None)
    run.tool_config.app_open_settle_delay = 0
    model = FakeModel()
    selector = DecisionSelector(
        model,
        "off",
        exclude={"get_language", "get_timezone", "list_apps", "swipe", "wait", "press_button"},
    )
    for _ in range(2):
        selection = await selector.select(run)
        assert selection.actions == [ToolCall("start_app", {"package": "example.clock"})]
        assert "Clock" in model.request.options[0].description
    call = selection.actions[0]
    result = await run.registry.by_name[call.name].acall(**call.parameters)
    assert result.raw_output.success and phone.launched == "example.clock"
    phone.list_apps.assert_awaited_once_with(include_system=True)


def test_decision_caps_and_places_options():
    model = FakeModel()
    request = selection_run(
        goal="Tap",
        tools=fake_tools("tap"),
        screenshot=b"image",
    )
    result = asyncio.run(DecisionSelector(model, "only").select(request))
    assert result.actions[0].parameters == {"x": 99, "y": 199}
    assert result.confidence == 0.9


def test_delegation_preserves_original_request_and_threshold():
    primary = StaticSelector(ActionSelection(actions=[ToolCall("click")], confidence=0.2))
    delegate = StaticSelector(ActionSelection(actions=[ToolCall("type", {"text": "hello"})]))
    request = selection_run(goal="Search", tools=fake_tools("type", "click"))
    result = asyncio.run(DelegatingSelector(primary, delegate).select(request))
    assert result.actions[0].name == "type"
    assert primary.requests[0] == request
    assert delegate.requests == [request]
    assert "delegate" not in request.registry.by_name


@pytest.mark.anyio
@pytest.mark.parametrize("vision", ["off", "on", "only"])
async def test_jev_vision_trims_its_own_request(vision):
    run = selection_run(
        "Tap",
        tools=fake_tools("tap_element", "tap", "wait"),
        elements=[{"index": 3, "text": "Clock", "bounds": "1,1,20,20"}],
    )
    run.device.screenshot.return_value = b"image"
    model = FakeModel()
    model.max_options = None
    await DecisionSelector(model, vision).select(run)
    request = model.request
    assert (request.state is None) == (vision == "only")
    assert (request.screenshot is None) == (vision == "off")
    assert run.device.screenshot.await_count == (vision != "off")
    names = {o.action.name for o in request.options}
    assert ("tap_element" in names) == (vision != "only")
    assert ("tap" in names) == (vision == "only")
    if vision == "only":
        assert names == {"tap", "wait"}
    assert set(run.registry.by_name) == {"tap_element", "tap", "wait"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "vision,elements,click,coordinate",
    [
        ("off", True, True, False),
        ("on", True, True, True),
        ("only", True, False, True),
        ("on", False, False, True),
    ],
)
async def test_llm_vision_trims_its_own_request(monkeypatch, vision, elements, click, coordinate):
    from llama_index.core.llms.mock import MockLLM

    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector

    png = BytesIO()
    Image.new("RGB", (100, 200)).save(png, "PNG")
    run = selection_run(
        "Tap",
        tools=fake_tools("tap_element", "tap", "wait"),
        elements=[{"index": 3}] if elements else [],
    )
    run.device.screenshot.return_value = png.getvalue()

    async def chat(self, messages, **kwargs):
        assert ("tap_element" in kwargs["tools"].by_name) == click
        assert ("tap" in kwargs["tools"].by_name) == coordinate
        assert (len(messages[-1].blocks) == 2) == coordinate
        assert ("<device_state>screen" in messages[-1].content) == (vision != "only")
        assert "wait" in kwargs["tools"].by_name

    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    await LlamaIndexLLMSelector(MockLLM(), vision, supports_native=True).select(run)
    assert run.device.screenshot.await_count == coordinate


@pytest.mark.anyio
async def test_jev_off_delegates_to_llm_on_before_fetching_screenshot(monkeypatch):
    from llama_index.core.llms.mock import MockLLM

    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector

    png = BytesIO()
    Image.new("RGB", (100, 200)).save(png, "PNG")
    run = selection_run("Tap", tools=fake_tools("tap", "delegate"))
    run.device.screenshot.return_value = png.getvalue()

    class Model(FakeModel):
        async def decide(self, request):
            assert request.screenshot is None and request.state is not None
            assert {option.action.name for option in request.options} == {"delegate"}
            run.device.screenshot.assert_not_awaited()
            return Decision(option_index=0)

    async def chat(self, messages, **kwargs):
        assert "tap" in kwargs["tools"].by_name
        assert len(messages[-1].blocks) == 2
        run.device.screenshot.assert_awaited_once()

    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    selector = DelegatingSelector(
        DecisionSelector(Model(), "off"),
        LlamaIndexLLMSelector(MockLLM(), "on", supports_native=True),
    )
    assert (await selector.select(run)).abstained


@pytest.mark.anyio
async def test_llm_rejects_a_hidden_tool_even_when_the_projected_registry_is_empty():
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            return ChatResponse(
                message=ChatMessage(
                    content='<function_calls><invoke name="tap"></invoke></function_calls>'
                )
            )

    run = selection_run("Tap", tools=fake_tools("tap"))
    assert (await Chat("off").select(run)).parse_error
    run.device.screenshot.assert_not_awaited()


def test_fallback_rejects_invalid_selection():
    primary = StaticSelector(ActionSelection(parse_error="Invalid tool call"))
    fallback = StaticSelector(ActionSelection(actions=[ToolCall("click")]))
    result = asyncio.run(
        FallbackSelector(primary, [fallback]).select(
            selection_run(goal="Tap", tools=fake_tools("click"))
        )
    )
    assert result.actions[0].name == "click"


@pytest.mark.anyio
async def test_jev_without_usable_tools_fails_instead_of_falling_back():
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.errors import SelectorCannotServe

    @tool
    async def set_proxy(run: Run, url: str) -> None:
        """Set the proxy URL."""

    class Model(FakeModel):
        async def decide(self, request):
            pytest.fail("Jev has no options to send")

    fallback = StaticSelector(ActionSelection(actions=[ToolCall("click_at")]))
    run = selection_run("Tap")
    run.registry = build_registry(run, [set_proxy], None).extend(
        fake_tools("tap"), {"complete", "wait"}
    )
    selector = FallbackSelector(DecisionSelector(Model(), "off"), [fallback])
    with pytest.raises(
        SelectorCannotServe, match=r"^Jev has no usable action among: set_proxy, tap\.$"
    ):
        await selector.select(run)
    assert fallback.requests == []
    run.device.screenshot.assert_not_awaited()


def test_delegated_models_prepare_independent_images_and_coordinate_spaces(monkeypatch):
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.ollama import Ollama

    from mobilerun_agents import Run
    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
    from mobilerun_agents.agent.action_selector.llm.models import Capabilities
    from mobilerun_agents.agent.tools.base import delegate
    from mobilerun_agents.agent.tools.registry import build_registry

    image = BytesIO()
    Image.new("RGB", (1440, 1920)).save(image, "PNG")
    run: Run = selection_run("tap center", screenshot=image.getvalue())
    run.device._supported_operations = frozenset({"tap"})
    run.state_provider.capabilities = set()
    run.ui.screen_width, run.ui.screen_height = 1440, 1920
    run.registry = build_registry(run, [delegate], None)
    views = []

    async def chat(self, messages, **kwargs):
        user = messages[-1]
        with Image.open(user.blocks[-1].resolve_image()) as sent:
            views.append((sent.size, user.content))
        body = (
            '<invoke name="delegate"></invoke>'
            if len(views) == 1
            else '<invoke name="tap"><parameter name="x">500</parameter><parameter name="y">500</parameter></invoke>'
        )
        return ChatResponse(message=ChatMessage(content=f"<function_calls>{body}</function_calls>"))

    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    selector = DelegatingSelector(
        LlamaIndexLLMSelector(Anthropic(model="claude-sonnet-4-5", api_key="test")),
        LlamaIndexLLMSelector(
            Ollama(model="gemma:latest"),
            image_limits=Capabilities(normalized_coords=True, vision=True),
        ),
    )
    before = run.state
    result = asyncio.run(selector.select(run))
    assert [size for size, _ in views] == [(952, 1270), (1440, 1920)]
    assert "952x1270" in views[0][1] and "0–1000" in views[1][1]
    assert result.actions[0].parameters == {"x": 720, "y": 960}
    assert run.state == before and run.capture.image == image.getvalue()
    run.device.screenshot.assert_awaited_once()


def test_llm_template_parse_override_can_call_super():
    class FakeChat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content='<function_calls><invoke name="wait"></invoke></function_calls>',
                )
            )

        def _parse(self, response):
            selection = super()._parse(response)
            selection.thought = "overridden"
            return selection

    result = asyncio.run(FakeChat().select(selection_run(goal="Wait")))
    assert result.actions[0].name == "wait"
    assert result.thought == "overridden"
    assert LLMSelector.__abstractmethods__ == frozenset({"_achat"})


@pytest.mark.parametrize("error", [ValueError("empty"), RuntimeError("transport")])
def test_transport_failures_propagate(error):
    class Chat(LLMSelector):
        async def _achat(self, messages, **kwargs):
            raise error

    with pytest.raises(type(error)) as raised:
        asyncio.run(Chat().select(selection_run(goal="Wait")))
    assert raised.value is error


@pytest.mark.parametrize("size", [(100, 2400), (2400, 1080), (100, 200), (280, 280)])
def test_typesafe_image_limits_and_coverage(size):
    source = BytesIO()
    Image.new("RGB", size, "red").save(source, "PNG")
    data, coverage = prepare_image(source.getvalue(), DecisionConfig())
    with Image.open(BytesIO(data)) as image:
        assert image.format == "JPEG"
        assert min(image.size) >= 280
        assert max(image.size) <= 1568
    assert all(0 < value <= 1 for value in coverage)


def test_jev_and_an_abstaining_delegate_share_the_run_registry_across_steps():
    base = fake_tools("tap_element", "tap")
    model = FakeModel()
    primary = DecisionSelector(model, "off")
    delegate = StaticSelector(None)
    composed = DelegatingSelector(primary, delegate, min_confidence=1.0)
    for index in (3, 5):
        request = selection_run(
            goal="Tap",
            tools=base,
            elements=[
                dict(
                    index=index,
                    bounds="0,0,20,20",
                    className="android.widget.Button",
                    text="tap",
                    isClickable=True,
                )
            ],
        )
        selection = asyncio.run(composed.select(request))
        assert model.request.registry is delegate.requests[-1].registry is request.registry
        assert [o.action for o in model.request.options] == [
            ToolCall("tap_element", {"index": index})
        ]
        assert selection.actions == [ToolCall("tap_element", {"index": index})]

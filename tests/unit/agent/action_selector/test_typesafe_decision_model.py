import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from PIL import Image

import mobilerun_agents.agent.action_selector.decision_model.typesafe as typesafe_module
from mobilerun_agents.agent.action_selector.decision_model.base import (
    ActionCandidate,
    Decision,
    DecisionInput,
    DecisionSelector,
)
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.tools.base import tool
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import (
    AgentConfig,
    DecisionConfig,
    MobileConfig,
    ToolsConfig,
)
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import fake_tools


@pytest.fixture
def sdk(monkeypatch):
    client = AsyncMock()
    client.system_one.return_value = SimpleNamespace(
        choices={"action": SimpleNamespace(choice="action_0", confidence=0.9)}, boxes={}
    )
    close = AsyncMock()

    class Scoped:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            await close()

    factory = Mock(return_value=Scoped())
    monkeypatch.setattr(typesafe_module, "AsyncTypeSafeClient", factory)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-env")
    return client, close, factory


def request(**kwargs):
    options = kwargs.pop(
        "options", (ActionCandidate(ToolCall("click", {"index": 7}), "Click Battery"),)
    )
    return DecisionInput(goal="Battery", instructions="Choose", options=options, **kwargs)


def test_opaque_labels_intent_and_memory_reach_scoped_sdk(sdk, monkeypatch):
    client, close, factory = sdk
    model = TypeSafeDecisionModel(model="jev", base_url="https://profile.example")
    turn = request(intent="Read the price", memory='click({"index": 7})\nClicked')
    decision = asyncio.run(model.decide(turn))
    assert (decision.option_index, decision.confidence) == (0, 0.9)
    sent = client.system_one.call_args.kwargs
    assert sent["questions"]["action"].criteria["action_0"]["parameters"] == {"index": 7}
    assert sent["state"] == {"goal": "Battery", "memory": turn.memory, "intent": turn.intent}
    factory.assert_called_once_with(
        model="jev", api_key="fake-env", base_url="https://profile.example"
    )
    close.assert_awaited_once()
    monkeypatch.setenv("TYPESAFE_API_KEY", "rotated-fake")
    asyncio.run(model.decide(turn))
    assert factory.call_args.kwargs["api_key"] == "rotated-fake"


def test_schema_candidate_sends_tool_description_as_context(sdk):
    @tool
    async def scroll(direction: Literal["down"]) -> None:
        """Scroll: down swipes the finger up."""

    client, _, _ = sdk
    turn = Turn("Scroll")
    turn.registry = build_registry(turn.device, [scroll], ToolsConfig())
    asyncio.run(
        turn.select(DecisionSelector(TypeSafeDecisionModel(), "off", exclude={"complete", "wait"}))
    )
    criteria = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert criteria["action_0"]["description"] == "scroll down"
    assert "down swipes the finger up" in criteria["action_0"]["context"]


@pytest.mark.parametrize("choice,confidence", [("abstain", 0.9), ("action_0", 0.2)])
def test_abstention_and_low_confidence(sdk, choice, confidence):
    client, close, _ = sdk
    client.system_one.return_value.choices["action"] = SimpleNamespace(
        choice=choice, confidence=confidence
    )
    turn = Turn()
    selection = asyncio.run(
        turn.select(DecisionSelector(TypeSafeDecisionModel(), "off", min_confidence=0.7))
    )
    assert selection.abstained and selection.confidence == confidence and not selection.calls
    assert any(getattr(event, "section", "") == "jev abstain" for event in turn.events)
    close.assert_awaited_once()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 2])
def test_invalid_confidence_is_validated_at_sdk_boundary(sdk, confidence):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].confidence = confidence
    with pytest.raises(RuntimeError, match=r"TypeSafe request failed \(ValidationError\)"):
        asyncio.run(TypeSafeDecisionModel().decide(request()))


@pytest.mark.parametrize("choice", ["action_99", None])
def test_unknown_or_missing_sdk_choice_is_repairable(sdk, choice):
    client, _, _ = sdk
    client.system_one.return_value.choices = (
        {} if choice is None else {"action": SimpleNamespace(choice=choice, confidence=0.9)}
    )
    config = DecisionConfig(
        invalid_arguments_prompt="assets/prompts/providers/empty_response.jinja2"
    )
    with pytest.raises(InvalidResponse, match="Empty response content"):
        asyncio.run(TypeSafeDecisionModel(config=config).decide(request()))


@pytest.mark.parametrize(
    "choice,success", [("complete_success", True), ("complete_failure", False)]
)
def test_completion_works_without_discrete_options_and_is_validated_at_sdk_boundary(
    sdk, choice, success
):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].choice = choice
    turn = Turn("Finish")
    selector = DecisionSelector(TypeSafeDecisionModel(), "off", exclude={"wait"})
    result = asyncio.run(turn.select(selector))
    assert result.calls[0].tool_kwargs["success"] is success
    assert result.calls[0].tool_kwargs["message"] is None
    criteria = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert criteria["complete_success"]["description"] == "complete: subgoal done"
    assert (
        criteria["complete_failure"]["description"]
        == "complete: report failure (blocked or impossible)"
    )
    with pytest.raises(InvalidResponse, match="Invalid arguments"):
        asyncio.run(TypeSafeDecisionModel().decide(request()))


def test_jev_always_offers_abstain_even_with_an_extra_option(sdk):
    client, _, _ = sdk
    decision = asyncio.run(
        TypeSafeDecisionModel().decide(
            request(options=(ActionCandidate(ToolCall("delegate"), "Delegate"),))
        )
    )
    assert decision.option_index == 0
    assert "abstain" in client.system_one.call_args.kwargs["questions"]["action"].criteria


def test_sdk_errors_close_client_and_hide_provider_body(sdk):
    client, close, _ = sdk
    client.system_one.side_effect = RuntimeError("private provider error")
    details = []
    with pytest.raises(RuntimeError, match=r"^TypeSafe request failed \(RuntimeError\)") as caught:
        asyncio.run(
            TypeSafeDecisionModel().decide(request(detail=lambda *entry: details.append(entry)))
        )
    assert "private provider error" not in repr(details)
    assert "private provider error" not in str(caught.value)
    close.assert_awaited_once()


@pytest.mark.timeout(5)
def test_cancelling_request_closes_scoped_client():
    async def scenario():
        started = asyncio.Event()
        closed = asyncio.Event()

        class PendingTransport(httpx2.AsyncBaseTransport):
            async def handle_async_request(self, request):
                started.set()
                await asyncio.Event().wait()

            async def aclose(self):
                closed.set()

        model = TypeSafeDecisionModel(api_key="fake", model="jev", transport=PendingTransport())
        task = asyncio.create_task(model.decide(request()))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(scenario())


def test_located_target_is_scaled_out_of_image_padding(sdk):
    client, _, _ = sdk
    output = BytesIO()
    Image.new("RGB", (100, 2400)).save(output, "PNG")
    client.system_one.return_value.boxes = {
        "target": SimpleNamespace(
            box=SimpleNamespace(left=0.05, right=0.15, top=0.2, bottom=0.4), confidence=0.9
        )
    }
    model = TypeSafeDecisionModel()
    result = asyncio.run(
        model.decide(
            request(
                screenshot=output.getvalue(),
                options=(ActionCandidate(ToolCall("tap"), "Tap", (100, 2400)),),
            )
        )
    )
    assert result.point == pytest.approx((0.1 / (65 / 280), 0.3))
    assert "target" in client.system_one.call_args.kwargs["questions"]
    asyncio.run(model.decide(request(screenshot=output.getvalue())))
    assert "target" not in client.system_one.call_args.kwargs["questions"]


def test_out_of_range_sdk_box_coordinate_is_repairable(sdk):
    client, _, _ = sdk
    output = BytesIO()
    Image.new("RGB", (100, 100)).save(output, "PNG")
    client.system_one.return_value.boxes = {
        "target": SimpleNamespace(
            box=SimpleNamespace(left=0.05, right=1.1, top=0.2, bottom=0.4), confidence=0.9
        )
    }
    with pytest.raises(InvalidResponse, match="Invalid arguments"):
        asyncio.run(
            TypeSafeDecisionModel().decide(
                request(
                    screenshot=output.getvalue(),
                    options=(ActionCandidate(ToolCall("tap"), "Tap", (100, 100)),),
                )
            )
        )


@dataclass
class Model:
    answer: Callable[[DecisionInput], Decision]
    locates_points: bool = False
    max_options = None

    async def decide(self, request: DecisionInput) -> Decision:
        return self.answer(request)


@pytest.mark.parametrize("index", [-1, 100])
@pytest.mark.anyio
async def test_out_of_range_option_is_repairable(index):
    config = DecisionConfig(
        invalid_arguments_prompt="assets/prompts/providers/empty_response.jinja2"
    )
    selector = DecisionSelector(Model(lambda _: Decision(option_index=index)), "off", config=config)

    with pytest.raises(InvalidResponse, match="Empty response content"):
        await Turn().select(selector)


@pytest.mark.parametrize("point", [None, (-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)])
@pytest.mark.anyio
async def test_bad_located_point_is_repairable(point):
    def answer(request):
        index = next(i for i, option in enumerate(request.options) if option.point_space)
        return Decision(option_index=index, point=point)

    selector = DecisionSelector(Model(answer, locates_points=True), "only")
    turn = Turn(tools=fake_tools("tap"), screenshot=b"fake screenshot")

    with pytest.raises(InvalidResponse, match="Invalid arguments"):
        await turn.select(selector)


@pytest.mark.parametrize(
    "explicit_config,explicit_confidence,abstains",
    [(None, None, True), (DecisionConfig(min_confidence=0.2), None, False), (None, 0.2, False)],
)
@pytest.mark.anyio
async def test_selection_uses_current_config_unless_explicit(
    explicit_config, explicit_confidence, abstains
):
    selector = DecisionSelector(
        Model(lambda _: Decision(option_index=0, confidence=0.6)),
        "off",
        config=explicit_config,
        min_confidence=explicit_confidence,
    )
    override = MobileConfig(agent=AgentConfig(decision=DecisionConfig(min_confidence=0.8)))

    with ConfigLoader.override(override):
        assert (await Turn().select(selector)).abstained is abstains


@pytest.mark.timeout(5)
@pytest.mark.anyio
async def test_selection_uses_config_timeout_unless_explicit():
    class SlowModel:
        max_options = None
        locates_points = False

        async def decide(self, request):
            await asyncio.sleep(0.05)
            return Decision(option_index=0)

    model = SlowModel()
    default = DecisionSelector(model, "off")
    explicit = DecisionSelector(model, "off", timeout=2)
    override = MobileConfig(agent=AgentConfig(decision=DecisionConfig(timeout=0.001)))

    with ConfigLoader.override(override):
        with pytest.raises(TimeoutError):
            await Turn().select(default)
        assert (await Turn().select(explicit)).calls[0].tool_name == "wait"

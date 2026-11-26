import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.config.user_config import TelemetryConfig
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel

from mobilerun_agents.agent.action_selector import ActionSelector, DelegatingSelector, Selection
from mobilerun_agents.agent.action_selector.decision_model.base import Decision, DecisionSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.operators import AgentOperator
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import MobileConfig, ToolsConfig
from mobilerun_agents.errors import SelectorCannotServe
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import MemoryDevice


def pick(name, **args):
    return ToolSelection(tool_id=name, tool_name=name, tool_kwargs=args)


class Stub(ActionSelector):
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = []

    async def select(self, ctx, device, memory, registry):
        self.calls.append(True)
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


class Model:
    max_options = None
    locates_points = False

    def __init__(self, *answers, completion_success: bool | None = None):
        self.answers = iter(answers)
        self.requests = []
        self.completion_success = completion_success

    async def decide(self, request):
        self.requests.append(request)
        name, confidence = next(self.answers)
        return Decision(
            option_index=next(
                i
                for i, item in enumerate(request.options)
                if item.action.name == name
                and (
                    self.completion_success is None
                    or item.action.parameters.get("success") is self.completion_success
                )
            ),
            confidence=confidence,
            thought="Jev considered the screen",
            message="Jev result",
        )


@pytest.mark.anyio
async def test_standalone_jev_does_not_offer_delegate():
    turn = Turn()
    model = Model(("wait", 0.9))
    assert (await turn.select(DecisionSelector(model, "off"))).calls[0].tool_name == "wait"
    assert "delegate" not in {candidate.action.name for candidate in model.requests[0].options}


@pytest.mark.anyio
async def test_delegate_retry_does_not_reask_primary():
    turn = Turn()
    model = Model(("delegate", 0.9))
    delegate = Stub(InvalidResponse("invalid"), Selection(calls=[pick("wait")]))
    selector = DelegatingSelector(DecisionSelector(model, "off"), delegate)
    with pytest.raises(InvalidResponse, match="invalid"):
        await selector.select(turn.ctx, turn.device, turn.memory, turn.registry)
    turn.state.repairs = 1
    restored = Turn()
    restored.state = turn.state.model_copy(deep=True)
    result = await restored.select(selector)
    assert result.calls == [pick("wait")]
    assert len(model.requests) == 1 and len(delegate.calls) == 2


@pytest.mark.anyio
async def test_low_confidence_abstains_even_when_jev_picked_delegate():
    turn = Turn()
    delegate = Stub()
    selector = DelegatingSelector(
        DecisionSelector(Model(("delegate", 0.2)), "off", min_confidence=0.7), delegate
    )
    result = await turn.select(selector)
    assert result.abstained and not result.calls and result.thought == "Jev considered the screen"
    assert not delegate.calls


@pytest.mark.anyio
async def test_explicit_option_calls_delegate_but_completion_is_jevs():
    turn = Turn()
    model = Model(("delegate", 0.9), ("complete", 0.8))
    model.max_options = 1
    delegate = Stub(Selection(calls=[pick("wait")], thought="The status page shows 42."))
    selector = DelegatingSelector(DecisionSelector(model, "off"), delegate)
    result = await selector.select(turn.ctx, turn.device, turn.memory, turn.registry)
    assert result.calls[0].tool_name == "wait"
    request = model.requests[0]
    option = next(candidate for candidate in request.options if candidate.action.name == "delegate")
    assert option.description == "hand this step to the LLM, e.g. to type text or open a URL"
    assert "delegate" not in request.registry and "delegate" not in turn.registry.by_name
    restored = Turn()
    restored.state = turn.state.model_copy(deep=True)
    restored.observe(turn.state.current.ui)
    result = await restored.select(selector)
    assert result.calls[0].tool_kwargs["message"] == "Jev result"
    assert model.requests[1].intent == "The status page shows 42."
    assert len(delegate.calls) == 1


@pytest.mark.parametrize(
    "calls", [[pick("delegate", unexpected=True)], [pick("delegate"), pick("wait")]]
)
@pytest.mark.anyio
async def test_invalid_delegate_calls_do_not_route(calls):
    turn = Turn()

    class InvalidPrimary(DecisionSelector):
        async def select(self, ctx, device, memory, registry, **kwargs):
            return Selection(calls=calls)

    selector = DelegatingSelector(InvalidPrimary(Model()), Stub())
    with pytest.raises(InvalidResponse):
        await selector.select(turn.ctx, turn.device, turn.memory, turn.registry)


@pytest.mark.anyio
async def test_recursive_delegate_becomes_repair():
    turn = Turn()
    selector = DelegatingSelector(
        DecisionSelector(Model(("delegate", 0.9)), "off"), Stub(Selection(calls=[pick("delegate")]))
    )
    with pytest.raises(InvalidResponse):
        await selector.select(turn.ctx, turn.device, turn.memory, turn.registry)


class Output(BaseModel):
    count: int


@pytest.mark.anyio
async def test_typed_success_delegates_selection_and_repair():
    turn = Turn()
    turn.registry = build_registry(turn.device, [], ToolsConfig(), output_model=Output)
    model = Model(("complete", 0.9), completion_success=True)
    typed_completion = pick("complete", success=True, message=None, output={"count": 42})
    delegate = Stub(InvalidResponse("redo typed completion"), Selection(calls=[typed_completion]))
    selector = DelegatingSelector(DecisionSelector(model, "off"), delegate)

    with pytest.raises(InvalidResponse, match="redo typed completion"):
        await turn.select(selector)
    turn.state.repairs = 1
    selection = await turn.select(selector)

    assert selection.calls == [typed_completion]
    assert len(model.requests) == 1


@pytest.mark.parametrize("output_model,success", [(Output, False), (None, True)])
@pytest.mark.anyio
async def test_completion_stays_with_primary_when_it_needs_no_typed_output(output_model, success):
    turn = Turn()
    turn.registry = build_registry(turn.device, [], ToolsConfig(), output_model=output_model)
    model = Model(("complete", 0.9), completion_success=success)
    selector = DelegatingSelector(DecisionSelector(model, "off"), Stub())

    selection = await turn.select(selector)

    assert selection.calls[0].tool_name == "complete"
    assert selection.calls[0].tool_kwargs == {"success": success, "message": "Jev result"}


@pytest.mark.anyio
async def test_bare_selector_rejects_typed_output_before_asking_model():
    turn = Turn()
    turn.registry = build_registry(turn.device, [], ToolsConfig(), output_model=Output)
    model = Model(("wait", 0.9))

    with pytest.raises(SelectorCannotServe, match="typed output.*LLM selector or a delegate"):
        await turn.select(DecisionSelector(model, "off"))
    assert not model.requests


@pytest.mark.anyio
async def test_agent_operator_surfaces_bare_selector_typed_output_error():
    class Screen(StateProvider):
        async def get_state(self) -> UIState:
            return UIState([{"index": 1}], "screen", "", {}, 100, 200)

    device = MemoryDevice()
    device._set_state_provider(Screen(device))
    operator = AgentOperator(selector=DecisionSelector(Model(("wait", 0.9)), "off"))
    with ConfigLoader.override(MobileConfig(analytics=TelemetryConfig(enabled=False))):
        with pytest.raises(SelectorCannotServe, match="typed output.*LLM selector or a delegate"):
            await operator.execute(device, "Count the results", output=Output, timeout=5)

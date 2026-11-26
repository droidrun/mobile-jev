import asyncio
from types import SimpleNamespace

import pytest
from llama_index.core.llms import ChatMessage, ChatResponse
from pydantic import BaseModel

import mobilerun_agents.agent.action_selector.decision_model.typesafe as typesafe_module
from mobilerun_agents.agent.action_selector import (
    ActionSelection,
    ActionSelector,
    DecisionSelector,
    DelegatingSelector,
    FallbackSelector,
)
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import Usage
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.logs.diagnostics import Detail, sink
from mobilerun_agents.agent.tools.base import complete
from mobilerun_agents.agent.tools.registry import build_registry
from tests.unit.agent.selector_fixtures import selection_run
from tests.unit.device_fixtures import fake_tools


class Stub(ActionSelector):
    def __init__(self, selection):
        self.selection = selection
        self.calls = 0

    async def select(self, run):
        self.calls += 1
        return self.selection or ActionSelection(abstained=True)


def test_primary_custom_decision_prompt_reaches_delegate_path(tmp_path):
    from mobilerun_agents.agent.action_selector import Decision
    from mobilerun_agents.config.user_config import DecisionConfig

    prompt = tmp_path / "custom.jinja2"
    prompt.write_text("CUSTOM {{ goal }} {{ 'delegate' in available_tools }}")

    class Model:
        screenshot_max_side = None
        max_options = None
        locates_points = False

        async def decide(self, request):
            assert "CUSTOM Goal True" in request.instructions
            return Decision(
                option_index=next(
                    i
                    for i, option in enumerate(request.options)
                    if option.action.name == "delegate"
                )
            )

    delegate = Stub(ActionSelection(actions=[ToolCall("wait")]))
    primary = DecisionSelector(Model(), config=DecisionConfig(system_prompt=str(prompt)))
    result = asyncio.run(
        DelegatingSelector(primary, delegate).select(
            selection_run(
                goal="Goal",
                tools=fake_tools("wait", "delegate"),
            )
        )
    )
    assert result.actions[0].name == "wait"
    assert delegate.calls == 1


@pytest.mark.parametrize("protocol", ["xml", "native"])
def test_explicit_delegate_never_executes_virtual_tool(protocol):
    primary = Stub(ActionSelection(actions=[ToolCall("delegate")]))
    delegate = Stub(ActionSelection(actions=[ToolCall("wait")]))
    result = asyncio.run(
        DelegatingSelector(primary, delegate).select(
            selection_run(goal="Wait", tools=fake_tools("delegate"))
        )
    )
    assert result.actions[0].name == "wait"
    assert delegate.calls == 1


@pytest.mark.parametrize(
    "actions",
    [
        [ToolCall("delegate", {"text": "bad"})],
        [ToolCall("delegate"), ToolCall("wait")],
    ],
)
def test_malformed_delegation_is_not_executed(actions):
    delegate = Stub(ActionSelection(actions=[ToolCall("wait")]))
    result = asyncio.run(
        DelegatingSelector(Stub(ActionSelection(actions=actions)), delegate).select(
            selection_run(goal="Wait")
        )
    )
    assert result.failure
    assert not result.actions
    assert delegate.calls == 0


def test_low_confidence_primary_retained_if_delegate_abstains():
    original = ActionSelection(actions=[ToolCall("wait")], confidence=0.3)
    result = asyncio.run(
        DelegatingSelector(Stub(original), Stub(None)).select(selection_run(goal="Wait"))
    )
    assert result is original


@pytest.mark.parametrize("composition", [DelegatingSelector, lambda a, b: FallbackSelector(a, [b])])
def test_usage_includes_both_attempts(composition):
    usage = Usage(request_tokens=1, response_tokens=2, total_tokens=3, requests=1)
    primary = Stub(ActionSelection(actions=[ToolCall("delegate")], usage=usage, failure="unusable"))
    delegate = Stub(ActionSelection(actions=[ToolCall("wait")], usage=usage))
    result = asyncio.run(
        composition(primary, delegate).select(
            selection_run(goal="Wait", tools=fake_tools("wait", "delegate"))
        )
    )
    assert result.usage.total_tokens == 6
    assert result.usage.requests == 2


def test_unavailable_delegate_is_rejected_without_execution():
    delegate = Stub(ActionSelection(actions=[ToolCall("wait")]))
    result = asyncio.run(
        DelegatingSelector(Stub(ActionSelection(actions=[ToolCall("delegate")])), delegate).select(
            selection_run(goal="wait", tools=fake_tools("wait"))
        )
    )
    assert result.failure and not result.actions
    assert delegate.calls == 0


@pytest.fixture
def jev_completion(monkeypatch):
    def make(success):
        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def system_one(self, **kwargs):
                choice = "complete_success" if success else "complete_failure"
                assert choice in kwargs["questions"]["action"].criteria
                return SimpleNamespace(choices={"action": {"choice": choice, "confidence": 0.9}})

        monkeypatch.setattr(typesafe_module, "AsyncTypeSafeClient", Client)
        jev = DecisionSelector(TypeSafeDecisionModel(api_key="fake"), vision="off")
        jev.debug_label = "jev"
        return jev

    return make


class CompletionChat(LLMSelector):
    debug_label = "llm:fake"

    def __init__(self, invocation):
        super().__init__(vision="off")
        self.invocation = invocation
        self.requests = []

    async def _achat(self, messages, **kwargs):
        self.requests.append(messages)
        return ChatResponse(
            message=ChatMessage(
                role="assistant", content=f"<function_calls>{self.invocation}</function_calls>"
            )
        )


def completion_run(output=None):
    run = selection_run(goal="Read the battery percentage")
    run.state_provider.capabilities = set()
    run.registry = build_registry(run, (complete,), output)
    return run


@pytest.mark.parametrize("success", [True, False])
def test_jev_completion_delegates_answer_and_includes_success_hint(jev_completion, success):
    delegate = CompletionChat(
        '<invoke name="complete"><parameter name="success">true</parameter>'
        '<parameter name="message">Battery is at 42%.</parameter></invoke>'
    )
    run = completion_run()
    events = []
    token = sink.set(events.append)
    try:
        result = asyncio.run(DelegatingSelector(jev_completion(success), delegate).select(run))
    finally:
        sink.reset(token)

    assert not result.parse_error
    assert result.actions == [
        ToolCall("complete", {"success": True, "message": "Battery is at 42%."})
    ]
    (messages,) = delegate.requests
    hint = "true" if success else "false"
    assert any(
        f'"primary_completion_hint": {{"success": {hint}}}' in (m.content or "") for m in messages
    )
    assert "primary_completion_hint" not in run.registry.by_name["complete"].metadata.description
    assert any(
        event == Detail(section="selector", text="jev → llm:fake (completion message)")
        for event in events
    )


def test_delegate_can_continue_instead_of_completing(jev_completion):
    delegate = CompletionChat('<invoke name="wait"></invoke>')
    result = asyncio.run(
        DelegatingSelector(jev_completion(True), delegate).select(completion_run())
    )

    assert not result.parse_error
    assert [action.name for action in result.actions] == ["wait"]


def test_plain_llm_completion_does_not_call_delegate():
    primary = CompletionChat(
        '<invoke name="complete"><parameter name="success">true</parameter>'
        '<parameter name="message">Battery is at 42%.</parameter></invoke>'
    )
    delegate = Stub(None)
    result = asyncio.run(DelegatingSelector(primary, delegate).select(completion_run()))

    assert not result.parse_error
    assert result.actions[0].parameters["message"] == "Battery is at 42%."
    assert delegate.calls == 0


def test_jev_without_delegate_keeps_its_completion_template(jev_completion):
    result = asyncio.run(jev_completion(True).select(completion_run()))

    assert result.actions == [ToolCall("complete", {"success": True, "message": "Goal achieved."})]


def test_typed_completion_still_uses_delegate_output(jev_completion):
    class Battery(BaseModel):
        percentage: int

    delegate = CompletionChat(
        '<invoke name="complete"><parameter name="success">true</parameter>'
        '<parameter name="message">Battery is at 42%.</parameter>'
        '<parameter name="output">{"percentage": 42}</parameter></invoke>'
    )
    result = asyncio.run(
        DelegatingSelector(jev_completion(True), delegate).select(completion_run(Battery))
    )

    assert not result.parse_error
    assert result.actions[0].parameters["output"] == {"percentage": 42}
    assert len(delegate.requests) == 1

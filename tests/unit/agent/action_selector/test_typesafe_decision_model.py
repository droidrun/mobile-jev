import asyncio
import json
import socket
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote_plus

import click
import httpx2
import pytest
from mobilerun.config.user_config import TelemetryConfig
from mobilerun.core.state.ui_state import UIState
from PIL import Image
from typesafe_sdk import Box

import mobilerun_agents.agent.action_selector.decision_model.typesafe as typesafe_module
from mobilerun_agents import ToolRegistry
from mobilerun_agents.agent.action_selector import ActionCandidate, DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.base import DecisionInput
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.logs import ConsoleLog, configure_cli_logging
from mobilerun_agents.agent.logs.diagnostics import protect, sink
from mobilerun_agents.agent.operators._consumers import RunConsumers
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.agent.workflows.agent import AgentWorkflow
from mobilerun_agents.config.user_config import CheckpointsConfig, MobileConfig
from tests.unit.agent.selector_fixtures import selection_run
from tests.unit.device_fixtures import MemoryDevice, fake_tools


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("TypeSafe unit tests cannot open network sockets")

    for method in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, method, blocked)


@pytest.fixture
def sdk(monkeypatch):
    client = SimpleNamespace(
        system_one=AsyncMock(
            return_value=SimpleNamespace(
                choices={"action": SimpleNamespace(choice="action_0", confidence=0.9)},
                boxes={},
            )
        )
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
    values = dict(
        goal="Battery",
        registry={"click": fake_tools("click")[0]},
        instructions="Choose",
        options=(ActionCandidate(ToolCall("click", {"index": 7}), "Click Battery"),),
    )
    values.update(kwargs)
    return DecisionInput(**values)


def test_opaque_labels_preserve_identity_and_scoped_client(sdk):
    client, close, factory = sdk
    decision = asyncio.run(TypeSafeDecisionModel(model="jev", api_key="fake").decide(request()))
    assert decision.option_index == 0
    assert decision.confidence == 0.9
    criteria = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert criteria["action_0"]["parameters"] == {"index": 7}
    factory.assert_called_once_with(model="jev", api_key="fake")
    close.assert_awaited_once()


def test_explicit_typesafe_client_options(sdk):
    _, _, factory = sdk
    model = TypeSafeDecisionModel(model="jev", base_url="https://profile.example")

    asyncio.run(model.decide(request()))

    factory.assert_called_once_with(
        model="jev",
        base_url="https://profile.example",
        api_key="fake-env",
    )


def test_schema_candidate_sends_tool_description_as_context(sdk):
    from typing import Literal

    from mobilerun_agents import Run, tool

    @tool
    async def scroll(run: Run, direction: Literal["down"]) -> None:
        """Scroll: down swipes the finger up."""

    client, _, _ = sdk
    run = selection_run("Scroll")
    run.registry = build_registry(run, [scroll], None)
    asyncio.run(
        DecisionSelector(TypeSafeDecisionModel(), "off", exclude={"complete", "wait"}).select(run)
    )
    criteria = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert criteria["action_0"]["description"] == "scroll down"
    assert "down swipes the finger up" in criteria["action_0"]["context"]


def test_key_is_resolved_again_for_each_request(sdk, monkeypatch):
    _, _, factory = sdk
    model = TypeSafeDecisionModel()
    asyncio.run(model.decide(request()))
    factory.assert_called_with(api_key="fake-env")
    monkeypatch.setenv("TYPESAFE_API_KEY", "rotated-fake")
    asyncio.run(model.decide(request()))
    factory.assert_called_with(api_key="rotated-fake")


@pytest.mark.parametrize("choice,confidence", [("abstain", 0.9), ("action_0", 0.2)])
def test_abstention_and_low_confidence(sdk, choice, confidence):
    client, close, _ = sdk
    client.system_one.return_value.choices["action"] = SimpleNamespace(
        choice=choice, confidence=confidence
    )
    result = asyncio.run(TypeSafeDecisionModel(min_confidence=0.7).decide(request()))
    assert result is None
    close.assert_awaited_once()


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 2])
def test_invalid_confidence_is_validated_at_sdk_boundary(sdk, confidence):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].confidence = confidence
    with pytest.raises(RuntimeError, match=r"TypeSafe request failed \(ValidationError\)"):
        asyncio.run(TypeSafeDecisionModel().decide(request()))


@pytest.mark.parametrize(
    "choice,success", [("complete_success", True), ("complete_failure", False)]
)
def test_completion_is_offered_only_when_enabled(sdk, choice, success):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].choice = choice
    run = selection_run("Battery", fake_tools("click", "complete"))
    result = asyncio.run(DecisionSelector(TypeSafeDecisionModel(), "off").select(run))
    assert result.actions[0].parameters["success"] is success
    with pytest.raises(ValueError, match="unknown or missing"):
        asyncio.run(TypeSafeDecisionModel().decide(request()))


def test_sdk_errors_close_client_and_hide_provider_body_from_debug(sdk, capsys):
    from mobilerun.cli.base.errors import render_error

    client, close, _ = sdk
    client.system_one.side_effect = RuntimeError("fake transport failure")
    with pytest.raises(RuntimeError) as raised:
        asyncio.run(TypeSafeDecisionModel().decide(request()))
    close.assert_awaited_once()
    render_error(raised.value, debug=True)
    debug = capsys.readouterr().err
    assert "fake transport failure" not in debug
    assert "TypeSafe request failed (RuntimeError)" in debug
    assert " in _execute_mock_call" in debug


def test_jev_can_complete_without_any_discrete_device_options(sdk):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].choice = "complete_success"
    run = selection_run("Finish", fake_tools("complete"))
    selection = asyncio.run(DecisionSelector(TypeSafeDecisionModel(), "off").select(run))
    assert selection.actions[0].name == "complete"
    choices = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert choices["complete_success"]["description"] == "complete: subgoal done"
    assert (
        choices["complete_failure"]["description"]
        == "complete: report failure (blocked or impossible)"
    )


def test_jev_with_delegate_does_not_offer_abstain(sdk):
    client, _, _ = sdk
    run = selection_run("Delegate", fake_tools("delegate"))
    result = asyncio.run(DecisionSelector(TypeSafeDecisionModel(), "off").select(run))
    assert result.actions == [ToolCall("delegate")]
    choices = client.system_one.call_args.kwargs["questions"]["action"].criteria
    assert "abstain" not in choices
    assert choices["action_0"]["description"] == (
        "delegate: hand this step to the LLM (typing text, or nothing listed fits)"
    )


def test_typesafe_wire_state_includes_current_delegate_intent(sdk):
    client, _, _ = sdk
    turn = request()
    turn.intent = "Read the price from the first listing"
    asyncio.run(TypeSafeDecisionModel(api_key="fake").decide(turn))
    assert client.system_one.call_args.kwargs["state"]["intent"] == turn.intent


def test_request_contains_memory_without_a_separate_history(sdk):
    client, _, _ = sdk
    memory = 'click({"index": 7})\n7 · click [7] ✓\nClicked'
    asyncio.run(TypeSafeDecisionModel().decide(request(memory=memory)))
    state = client.system_one.call_args.kwargs["state"]
    assert state["memory"] == memory
    assert "history" not in state


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
            box=SimpleNamespace(left=0.05, right=0.15, top=0.2, bottom=0.4),
            confidence=0.9,
        )
    }
    result = asyncio.run(
        TypeSafeDecisionModel().decide(
            request(
                screenshot=output.getvalue(),
                registry={"click_at": fake_tools("click_at")[0]},
                options=(ActionCandidate(ToolCall("click_at"), "Tap", (100, 2400)),),
            )
        )
    )
    assert result.point == pytest.approx((0.1 / (65 / 280), 0.3))
    assert "target" in client.system_one.call_args.kwargs["questions"]


def test_image_without_point_options_does_not_request_location(sdk):
    client, _, _ = sdk
    output = BytesIO()
    Image.new("RGB", (280, 280)).save(output, "PNG")
    asyncio.run(TypeSafeDecisionModel().decide(request(screenshot=output.getvalue())))
    assert "target" not in client.system_one.call_args.kwargs["questions"]


@pytest.fixture
def jev_console():
    @contextmanager
    def console(debug):
        # Keep the sink attached even in normal mode to exercise the console gate.
        with click.Context(click.Command("agent")):
            configure_cli_logging(debug)
            token = sink.set(ConsoleLog().on_event)
            try:
                yield
            finally:
                sink.reset(token)

    return console


@pytest.mark.parametrize(
    "vision,input_line,groups,answer",
    [
        (
            "off",
            "tree (6 chars) · no image · no locate",
            "1 · tap_element 1",
            '1 tap_element [12] "Battery"',
        ),
        (
            "on",
            "tree (6 chars) · image 280×280 · no locate",
            "1 · tap_element 1",
            '1 tap_element [12] "Battery"',
        ),
        ("only", "image 280×280 · jev locate: requested", "1 · tap 1", "tap on screen"),
    ],
)
def test_jev_debug_reports_actual_vision_inputs(
    sdk, jev_console, capsys, vision, input_line, groups, answer
):
    client, _, _ = sdk
    client.system_one.return_value.boxes = {
        "target": SimpleNamespace(
            box=SimpleNamespace(left=0.2, right=0.4, top=0.4, bottom=0.6), confidence=0.9
        )
    }
    output = BytesIO()
    Image.new("RGB", (280, 280)).save(output, "PNG")
    run = selection_run(
        "Battery",
        screenshot=output.getvalue(),
        elements=[{"index": 12, "text": "Battery", "type": "Button", "bounds": "20,20,50,50"}],
    )
    taps = []

    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            taps.append((x, y))

    provider = run.state_provider
    provider.requires_coordinate_tools = False
    provider.capabilities = {"element_index"}
    phone = Phone(supported_operations=frozenset({"tap"}))
    phone._set_state_provider(provider)
    phone.screenshot = run.device.screenshot
    run.device = phone
    tools = build_registry(run, [], None)
    run.registry = ToolRegistry(
        run, tuple(tools.by_name[name] for name in ("tap", "tap_element", "complete"))
    )
    selector = DecisionSelector(TypeSafeDecisionModel(min_confidence=0.7), vision)

    async def select():
        selection = await selector.select(run)
        if vision == "only":
            call = selection.actions[0]
            await run.registry.by_name[call.name].acall(**call.parameters)
        return selection

    with jev_console(True):
        selection = asyncio.run(select())
    sent = client.system_one.call_args.kwargs
    assert ("ui" in sent["state"]) is (vision != "only")
    assert ("screenshot" in sent["state"]) is (vision != "off")
    assert ("target" in sent["questions"]) is (vision == "only")
    choices = sent["questions"]["action"].criteria
    assert [choice["tool"] for choice in choices.values() if isinstance(choice, dict)] == [
        "tap" if vision == "only" else "tap_element",
        "complete",
        "complete",
    ]
    client.system_one.assert_awaited_once()
    log = capsys.readouterr().err
    assert f"jev input: {input_line}" in log
    assert f"jev choices: 3 · complete 2, {groups.split(' · ')[1]} · +abstain" in log
    assert "jev option:" not in log
    assert f"jev answer: {answer}" in log
    assert "confidence=0.9 (min=0.7)" in log
    assert "latency=" in log
    assert "image: no image" not in log
    assert "locate unused" not in log
    if vision == "only":
        assert type(sent["questions"]["target"]) is Box
        assert "box (56,112,112,168) in image px → tap (30,100) device px" in " ".join(log.split())
        assert taps == [(30, 100)]
    assert selection.actions[0].name == ("tap" if vision == "only" else "tap_element")


@pytest.mark.parametrize(
    "choice,confidence,reason",
    [
        ("abstain", 0.9, "model chose abstain"),
        ("action_0", 0.2, "confidence 0.2 < 0.7"),
    ],
)
def test_jev_debug_reports_abstention_reason(sdk, jev_console, capsys, choice, confidence, reason):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"] = SimpleNamespace(
        choice=choice, confidence=confidence
    )
    with jev_console(True):
        assert asyncio.run(TypeSafeDecisionModel(min_confidence=0.7).decide(request())) is None
    log = capsys.readouterr().err
    assert f"jev abstain: {reason}" in log
    assert "jev answer:" in log


def test_jev_debug_groups_choices_and_masks_selected_label(sdk, jev_console, capsys):
    client, _, _ = sdk
    client.system_one.return_value.choices["action"].choice = "action_0"
    options = (
        ActionCandidate(ToolCall("click", {"index": 12}), 'Click button "fake-env protected"'),
        ActionCandidate(ToolCall("click", {"index": 13}), 'Click text field "editable value"'),
        ActionCandidate(ToolCall("click", {"index": 14}), 'Click button "token=unmarked"'),
        *(
            ActionCandidate(ToolCall("click", {"index": i}), f"Click Item {i}")
            for i in range(15, 30)
        ),
    )
    state = UIState(
        [{"index": 13, "type": "TextField", "details": {"editable": True}}],
        "tree body not for logs",
        "focused body not for logs",
        {},
        100,
        200,
    )
    with jev_console(True):
        protect("protected")
        assert asyncio.run(TypeSafeDecisionModel().decide(request(state=state, options=options)))
    log = capsys.readouterr().err
    assert "jev choices: 18 · click 18 · +abstain" in log
    assert "jev option:" not in log
    assert '1 click [12] "*** ***"' in log
    assert "(min=" not in log
    for private in ("fake-env", "protected", "unmarked", "editable value", "body not for logs"):
        assert private not in log


def test_decision_sidecar_preserves_public_parameters_and_masks_declared_secrets():
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.action_selector.decision_model.typesafe import _recorded_choices
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import tool_run

    @tool(sensitive_parameters=frozenset({"token"}))
    async def mixed(run: Run, package: str, token: str) -> None:
        """Use a package and token."""

    run = tool_run()
    req = DecisionInput("test", registry=build_registry(run, [mixed], None).by_name)
    params = [
        {"package": "example.app"},
        {"x1": 12, "y1": 35, "x2": 7, "y2": 2},
        {"package": "example.app", "token": "fixture-value"},
    ]
    choices = tuple(
        ActionCandidate(ToolCall(name, args), name)
        for name, args in zip(("start_app", "swipe", "mixed"), params, strict=True)
    )
    rows = _recorded_choices(req, choices, False)
    assert [row["tool_call"]["parameters"] for row in rows] == [
        *params[:2],
        {"package": "example.app", "token": "***"},
    ]


def test_jev_normal_mode_omits_debug_output(sdk, jev_console, capsys):
    with jev_console(False):
        decision = asyncio.run(TypeSafeDecisionModel().decide(request()))
    assert decision.option_index == 0
    assert capsys.readouterr().err == ""


def test_jev_debug_does_not_log_provider_error_text(sdk, jev_console, capsys):
    client, _, _ = sdk
    client.system_one.side_effect = RuntimeError("raw provider body")
    with jev_console(True), pytest.raises(RuntimeError):
        asyncio.run(TypeSafeDecisionModel().decide(request()))
    log = capsys.readouterr().err
    assert "jev answer: unavailable" in log
    assert "raw provider body" not in log


@pytest.mark.parametrize("enabled,debug", [(True, True), (True, False), (False, True)])
def test_jev_records_full_masked_decisions_only_with_checkpoints(
    sdk, capsys, tmp_path, enabled, debug, monkeypatch
):
    from tests.unit.agent.workflows.test_run import resources

    client, _, _ = sdk
    client.system_one.return_value.choices["action"].choice = "abstain"
    config = MobileConfig(
        checkpoints=CheckpointsConfig(enabled=enabled),
        analytics=TelemetryConfig(enabled=False),
    )
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path))

    async def run():
        args = await resources()
        ui = args["device"].state._resolved_provider.get_state.return_value
        ui.elements[0].update(text="fixture-cell-value", type="TextField", clickable=True)
        ui.elements += [
            {
                "index": i,
                "text": f"Button {i}",
                "type": "Button",
                "bounds": f"{i},0,{i+2},8",
                "clickable": True,
            }
            for i in range(20, 40)
        ]
        ui.elements[-1]["text"] = "fake-env token=unmarked"
        flow = AgentWorkflow(DecisionSelector(TypeSafeDecisionModel(), "off"))
        async with RunConsumers(config, "decision-fixture", 1) as consumers:
            await consumers.consume(flow.run(**args, max_steps=1))

    with click.Context(click.Command("agent")):
        configure_cli_logging(debug)
        asyncio.run(run())
    log = capsys.readouterr().err
    path = tmp_path / "decision-fixture" / "decisions" / "0001.json"
    assert path.exists() is enabled
    assert ("(full list: decisions/0001.json)" in " ".join(log.split())) is (enabled and debug)
    if enabled:
        payload = path.read_text()
        for private in ("fixture-cell-value", "fake-env", "unmarked"):
            assert private not in payload
        choices = json.loads(payload)["choices"]
        taps = [c for c in choices if c["tool_call"]["name"] == "tap_element"]
        assert len(taps) == 21
        assert taps[0]["tool_call"] == {"name": "tap_element", "parameters": {"index": 3}}
        assert 'tap_element [3] "***"' in taps[0]["label"]


@pytest.mark.parametrize("debug", [False, True])
def test_jev_sidecar_masks_tracked_values_in_ordinary_arguments(sdk, tmp_path, debug, monkeypatch):
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.action_selector.base import ActionSelection, ActionSelector
    from tests.unit.agent.workflows.test_run import complete, resources

    private = 'fixture "é"\n  marker / value'
    provider = "fixture / provider"
    private_packages = (
        private,
        json.dumps(private)[1:-1],
        repr(private)[1:-1],
        " ".join(private.split()),
        quote_plus(private),
        provider,
        quote_plus(provider),
    )

    @tool(sensitive_parameters=frozenset({"token"}))
    async def authorize(run: Run, token: str) -> None:
        """Use a fixture credential."""

    class Selector(ActionSelector):
        async def select(self, run: Run) -> ActionSelection:
            if not run.state.actions:
                return ActionSelection(actions=[ToolCall("authorize", {"token": private})])
            options = tuple(
                ActionCandidate(ToolCall("start_app", {"package": package}), "Open app")
                for package in (*private_packages, "example.app")
            ) + (
                ActionCandidate(ToolCall("tap", {"x": 12, "y": 35}), "Tap"),
            )
            await TypeSafeDecisionModel(api_key=provider).decide(request(options=options))
            return ActionSelection(actions=[complete()])

    config = MobileConfig(
        checkpoints=CheckpointsConfig(),
        analytics=TelemetryConfig(enabled=False),
    )
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path))
    config.tracing.enabled = False

    async def run():
        async with RunConsumers(config, "tracked-values", 2) as consumers:
            await consumers.consume(
                AgentWorkflow(Selector()).run(**await resources(), tools=[authorize])
            )

    with click.Context(click.Command("agent")):
        configure_cli_logging(debug)
        asyncio.run(run())

    sidecar = tmp_path / "tracked-values" / "decisions" / "0001.json"
    calls = [row["tool_call"] for row in json.loads(sidecar.read_text())["choices"]]
    packages = [call["parameters"]["package"] for call in calls if call["name"] == "start_app"]
    if packages[:-1] != ["***"] * len(private_packages):
        pytest.fail("Sidecar did not mask every tracked value", pytrace=False)
    assert packages[-1] == "example.app"
    assert {"name": "tap", "parameters": {"x": 12, "y": 35}} in calls

import asyncio
import json
import logging
import socket
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock, NonCallableMock

import pytest
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from mobilerun.core.extensions import Extension, action
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.settings import get_macro_settings
from pydantic import BaseModel
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents.agent.action_selector import DelegatingSelector
from mobilerun_agents.agent.action_selector.base import ActionSelection, ActionSelector
from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.action_selector.llm.parsing import ToolCall
from mobilerun_agents.agent.operators import AgentOperator
from mobilerun_agents.agent.tools.base import complete
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionOutcome, ActionRecord, AgentState
from mobilerun_agents.agent.workflows.events import Act, Invalid, Observed, StepFinished
from mobilerun_agents.config.settings import get_agent_settings
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig
from tests.unit.agent.selector_fixtures import captured_turn
from tests.unit.device_fixtures import MemoryDevice


class Answer(BaseModel):
    count: int


def recorded_macro(run_folder: str) -> Path:
    return Path(get_macro_settings().macro_directory) / Path(run_folder).name / "macro.json"


def test_public_operator_choices():
    from inspect import Parameter, signature

    from mobilerun_agents import AgentOperator, ExternalOperator, PlanningOperator, agent

    for cls, names in (
        (AgentOperator, ("selector", "tools")),
        (PlanningOperator, ("planner", "executor", "tools")),
        (ExternalOperator, ("plugin",)),
    ):
        assert getattr(agent, cls.__name__) is cls
        params = signature(cls).parameters
        assert tuple(params) == names
        assert all(p.kind == Parameter.POSITIONAL_OR_KEYWORD for p in params.values())


class Selector(ActionSelector):
    async def select(self, run):
        assert run.registry.by_name["complete"].metadata.get_parameters_dict()["properties"][
            "output"
        ]["properties"]["count"]
        return ActionSelection(
            actions=[
                ToolCall("complete", {"message": "done", "success": True, "output": {"count": 3}})
            ]
        )


class Extra(Extension):
    default_name = "extra"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def block_sockets(monkeypatch):
    def blocked(*args):
        pytest.fail("Operator tests cannot connect to a network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class Scripted(Workflow):
    action_selector = None

    @step
    async def script(self, ctx: Context[AgentState], ev: StartEvent) -> StopEvent:
        self.bindings = set(ev.device.extensions)
        self.run_tools = {t.function.__name__ for t in ev.tools}
        if ev.goal == "wait":
            self.entered.set()
            try:
                await asyncio.Future()
            finally:
                self.stopped.set()
        snapshot = StepSnapshot(
            goal=ev.goal,
            step=1,
            ui=UIState([], "", "", {}, 0, 0),
            device_context=[],
        )
        record = ActionRecord(
            step=1,
            name="complete",
            args={},
            ok=True,
            summary="finished",
            before=snapshot,
            screenshot=None,
            ui_state=None,
        )
        outcome = ActionOutcome.model_validate(record.model_dump(exclude={"before"}))
        for event in [
            Invalid(error="repair"),
            Observed(step=1, package=None, activity=None),
            Act(selection=ActionSelection(actions=[ToolCall("complete")])),
            StepFinished(action=outcome, display_action=outcome),
        ]:
            ctx.write_event_to_stream(event)
        await ctx.store.set_state(AgentState(goal=ev.goal, step=1, actions=[record]))
        return StopEvent(
            result=AgentResult(
                success=True, answer="done", reason="complete", output=Answer(count=3), steps=1
            )
        )


class Tappable(MemoryDevice):
    async def tap(self, x: int, y: int) -> None:
        await self.tapped(x, y)


@pytest.fixture
def phone():
    device = Tappable()
    provider = NonCallableMock(
        spec=StateProvider, capabilities=set(), requires_coordinate_tools=False
    )
    provider.get_state = AsyncMock(return_value=UIState([], "screen", "", {}, 100, 100))
    device._set_state_provider(provider)
    device.screenshot = AsyncMock(return_value=b"image")
    return device


@pytest.fixture
def config(tmp_path, monkeypatch):
    cfg = MobileConfig.model_validate({"analytics": {"enabled": False}})
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path / "runs"))

    monkeypatch.setattr("mobilerun_agents.config.loader.ConfigLoader.load", lambda *_: cfg)
    return cfg


@pytest.mark.anyio
async def test_default_operator_uses_real_selector_factory(phone, tmp_path, monkeypatch):
    class FakeGemini(MockLLM):
        def __init__(self, model, api_key, http_options, **adapter_options):
            super().__init__()
            assert model == "gemini-3.8-flash"
            assert "temperature" not in adapter_options
            assert api_key == "unit-key" and http_options.retry_options.attempts == 1

        async def astream_chat(self, messages, **kwargs):
            assert any("count items" in (message.content or "") for message in messages[:-1])

            async def chunks():
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content=(
                            '<function_calls><invoke name="complete">'
                            '<parameter name="success">true</parameter>'
                            '<parameter name="message">done</parameter></invoke></function_calls>'
                        ),
                    )
                )

            return chunks()

    monkeypatch.chdir(tmp_path)
    from io import BytesIO

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (20, 20)).save(image, format="PNG")
    phone.screenshot.return_value = image.getvalue()
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-key")
    monkeypatch.setattr("mobilerun_agents.agent.analytics.consumer._client", Mock())
    monkeypatch.setitem(MODELS, "gemini", replace(MODELS["gemini"], adapter=FakeGemini))
    operator = AgentOperator()
    result = await operator.execute(phone, "count items")
    assert (result.success, result.answer, result.steps) == (True, "done", 1)
    assert not phone.extensions


@pytest.mark.anyio
@pytest.mark.parametrize("tools,expected", [((), set()), ([], set())])
async def test_setup_default_and_delegation_tools(phone, config, monkeypatch, tools, expected):
    import mobilerun_agents.agent.operators.agent as module
    from mobilerun_agents.agent.action_selector.loader import SetupSpec

    config.checkpoints.enabled = False
    loader = AsyncMock(return_value=SetupSpec(selector=DelegatingSelector(Selector(), Selector())))
    monkeypatch.setattr(module, "default_setup", loader)
    flow = Scripted()
    monkeypatch.setattr(module, "AgentWorkflow", lambda **_: flow)
    result = await AgentOperator(tools=tools).execute(phone, "count")
    loader.assert_awaited_once_with()
    assert flow.bindings == expected and not phone.extensions
    assert flow.run_tools == {"delegate"}
    assert result.trajectory_path is None and not get_agent_settings().runs_dir.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("override", [False, True])
async def test_recording_uses_base_stores(phone, config, tmp_path, monkeypatch, override):
    from click.testing import CliRunner
    from mobilerun.macro.cli.list import list as list_macros

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    base_config = tmp_path / "base-config"
    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(base_config))
    macro_root = tmp_path / "shared-macros" if override else base_config / "macros"
    if override:
        monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(macro_root))
    else:
        monkeypatch.delenv("MOBILERUN_AGENT_RUNS_DIR")

    result = await AgentOperator(selector=Selector()).execute(phone, "count", output=Answer)

    folder = Path(result.trajectory_path)
    assert folder.parent == (tmp_path / "runs" if override else base_config / "runs")
    assert (folder / "checkpoints/0001.json").is_file()
    assert (folder / "ui_states/0001.json").is_file()
    assert not (folder / "macro.json").exists()
    assert not (cwd / "runs").exists()
    assert (macro_root / folder.name / "macro.json").is_file()
    listed = CliRunner().invoke(list_macros, ["--json"])
    assert listed.exit_code == 0, listed.exception
    assert json.loads(listed.output) == [
        {"name": folder.name, "description": "count", "actions": 0}
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("capture", [False, True])
async def test_operator_records_maps_and_leaves_caller_binding(phone, config, capture):
    class CaptureSelector(Selector):
        async def select(self, run):
            if capture:
                assert await run.screenshot() == await run.screenshot() == b"image"
            return await super().select(run)

    await Extra().bind(phone)
    before = phone.actions.list()
    op = AgentOperator(selector=CaptureSelector())
    result = await op.execute(phone, "count", output=Answer)
    assert (result.success, result.answer, result.steps, result.device, result.output) == (
        True,
        "done",
        1,
        "memory",
        Answer(count=3),
    )
    folder = Path(result.trajectory_path)
    image = get_agent_settings().images_dir / folder.name / "0001.png"
    assert image.exists() == capture
    if capture:
        assert image.read_bytes() == b"image"
        checkpoint = json.loads((folder / "checkpoints/0001.json").read_text())
        assert checkpoint["actions"][0]["screenshot"] == str(image)
        assert list(image.parent.iterdir()) == [image]
    assert not (folder / "screenshots").exists()
    assert phone.screenshot.await_count == int(capture)
    assert recorded_macro(result.trajectory_path).is_file()
    assert list(phone.extensions) == ["extra"]
    assert phone.actions.list() == before


@pytest.mark.anyio
async def test_scripted_stream_notifies_consumers(phone, config, monkeypatch, caplog):
    import mobilerun_agents.agent.operators._consumers as module

    analytics = Mock(finish=AsyncMock())
    console = Mock(wraps=module.ConsoleLog())
    monkeypatch.setattr(module, "ConsoleLog", lambda: console)
    monkeypatch.setattr(module, "RunAnalytics", lambda *_: analytics)
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    monkeypatch.setattr("mobilerun_agents.agent.operators.agent.AgentWorkflow", Scripted)
    op = AgentOperator(selector=Selector())
    result = await op.execute(phone, "count", output=Answer)
    assert result.output == Answer(count=3) and not phone.extensions
    assert [m[0] for m in analytics.method_calls] == ["on_observe", "finish"]
    assert any(isinstance(call.args[0], Act) for call in console.on_event.call_args_list)
    assert [r.message for r in caplog.records if r.name.endswith("logs.console")] == [
        "Retrying invalid selection",
        "1 · complete ✓",
        f"✓ done (1 step · {result.trajectory_path})",
    ]
    assert recorded_macro(result.trajectory_path).exists()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [True, False])
async def test_interrupted_run_releases_tools(phone, config, cancel, monkeypatch, caplog):
    entered, stopped = asyncio.Event(), asyncio.Event()
    flow = Scripted(timeout=5 if cancel else 0.5)
    flow.entered, flow.stopped = entered, stopped
    monkeypatch.setattr("mobilerun_agents.agent.operators.agent.AgentWorkflow", lambda **_: flow)
    op = AgentOperator(selector=Selector())
    task = asyncio.create_task(op.execute(phone, "wait", timeout=5 if cancel else 0.5))
    await asyncio.wait_for(entered.wait(), 4)
    if cancel:
        caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert stopped.is_set() and not phone.extensions
    if cancel:
        assert "✗ cancelled (0 steps · " in caplog.text
        assert str(get_agent_settings().runs_dir) in caplog.text


class Choices(ActionSelector):
    def __init__(self, *calls):
        self.calls = iter(calls)
        self.requests = []

    async def select(self, run):
        self.requests.append(await captured_turn(run))
        return ActionSelection(actions=[next(self.calls)])


@pytest.mark.anyio
async def test_focus_tap_is_replayable_when_typing_fails(phone, config):
    phone._supported_operations = {"input_text", "tap"}
    phone.tapped = AsyncMock()
    provider = phone.state._resolved_provider
    provider.capabilities = {"direct_text_input", "element_index"}
    provider.get_state.return_value = UIState(
        [{"index": 1, "bounds": "0,0,20,40"}], "", "", {}, 100, 100
    )
    phone._input_text_raw = AsyncMock(return_value=False)
    result = await AgentOperator(
        selector=Choices(
            ToolCall("tap_element", {"index": 1}), ToolCall("input_text", {"text": "fixture"})
        )
    ).execute(phone, "type", max_steps=2)
    phone.tapped.assert_awaited_once_with(10, 20)
    actions = json.loads(recorded_macro(result.trajectory_path).read_text())["actions"]
    assert [a["action_type"] for a in actions] == ["tap"]


@pytest.mark.anyio
@pytest.mark.parametrize("fails", [True, False])
async def test_attempted_typing_and_primitive_guards(phone, config, fails):
    phone._supported_operations = {"input_text", "tap"}
    phone.tapped = AsyncMock()
    provider = phone.state._resolved_provider
    provider.capabilities = {"direct_text_input", "element_index"}

    def screen(focused):
        return UIState(
            [{"index": 1, "bounds": "0,0,20,40", "focused": focused}], "", "", {}, 100, 100
        )

    provider.get_state.side_effect = [screen(False), screen(False), screen(True)]
    phone._input_text_raw = AsyncMock(return_value=not fails)
    call = ToolCall("input_text", {"text": "fixture-literal"})
    result = await AgentOperator(selector=Choices(call)).execute(
        phone, "enter fixture-literal", max_steps=1
    )
    folder = Path(result.trajectory_path)
    assert "fixture-literal" not in (folder / "checkpoints/0001.json").read_text()


@pytest.mark.anyio
async def test_extension_macro_replays_public_call(phone, config, caplog):
    from mobilerun.macro.player import MacroPlayer

    class Echo(Extension):
        default_name = "echo"

        @action
        async def say(self, value: str) -> None:
            """Remember the value."""
            phone.inputs.append(value)

        @action(sensitive_parameters=frozenset({"value"}))
        async def private(self, value: str) -> str:
            """Use a sensitive value."""
            return value

    await Echo().bind(phone)
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    result = await AgentOperator(
        selector=Choices(
            ToolCall("echo__say", {"value": "ok"}),
            ToolCall("echo__private", {"value": "fixture-private"}),
            ToolCall("echo__say", {"value": "after"}),
        ),
    ).execute(phone, "echo", max_steps=3)
    player = MacroPlayer(phone, delay_between_actions=0)
    macro = player.load_macro_from_file(recorded_macro(result.trajectory_path))
    assert "fixture-private" not in json.dumps(macro)
    assert "echo.private ***" in caplog.text and "fixture-private" not in caplog.text
    assert await player.replay_macro(macro)
    assert phone.inputs == ["ok", "after", "ok", "after"]


@pytest.mark.anyio
@pytest.mark.parametrize("encode", [str, lambda s: json.dumps(s)[1:-1], lambda s: repr(s)[1:-1]])
async def test_macro_masks_goal_and_drops_only_secret_echo_guards(phone, config, encode):
    from mobilerun.core.credentials import DictCredentials
    from mobilerun.macro.player import MacroPlayer

    private = "fixture-'yes'-\"quote\"\nmarker"
    provider = phone.state._resolved_provider
    safe_state = UIState(
        [{"index": 1, "text": "public", "className": "TextView"}],
        "",
        "",
        {"focusedElement": {"index": 1}, "isEditable": True},
        100,
        100,
    )
    echo_state = UIState(
        [{"index": 1, "text": encode(private), "className": "TextView"}], "", "", {}, 100, 100
    )
    provider.get_state.return_value = safe_state
    phone._supported_operations = {"input_text", "open_url"}
    await DictCredentials({"shop": private}).bind(phone)

    async def type_and_echo(text, *args, **kwargs):
        assert text == private
        provider.get_state.return_value = echo_state
        return True

    async def open_url(url: str) -> None:
        assert url == "https://example.invalid/"

    phone._input_text_raw = type_and_echo
    phone.open_url = open_url
    public = ToolCall("open_url", {"url": "https://example.invalid/"})
    vault = ToolCall("vault__type_field", {"package": None, "field": "password", "name": "shop"})
    result = await AgentOperator(Choices(public, vault, public)).execute(
        phone, f"Type {encode(private)}", max_steps=3
    )
    path = recorded_macro(result.trajectory_path)
    macro = json.loads(path.read_text())
    assert macro["description"] == "Type ***"
    first, credential, last = macro["actions"]
    assert first["pre_state"] is not None
    assert credential["arguments"]["name"] == "shop"
    assert last["pre_state"] is None
    provider.get_state.return_value = safe_state
    player = MacroPlayer(phone, delay_between_actions=0, state_threshold=1, state_timeout=0)
    assert await player.replay_macro(player.load_macro_from_file(path))
    assert player.executed_steps == 3


@pytest.mark.anyio
async def test_real_repair_is_reported(phone, config, caplog):
    caplog.set_level(logging.INFO)
    selector = Choices(
        ToolCall("complete", {"success": "not-bool", "message": "bad"}),
        ToolCall("complete", {"success": True, "message": "ok"}),
    )
    result = await AgentOperator(selector=selector).execute(phone, "repair")
    assert result.answer == "ok"
    assert [r.message for r in caplog.records].count("Retrying invalid selection") == 1


@pytest.mark.anyio
async def test_notes_reach_next_request_as_run_notes(phone, config):
    class Notes(Choices):
        async def select(self, run):
            chosen = await super().select(run)
            chosen.memory_update = "row seven"
            return chosen

    phone._supported_operations = {"tap"}
    phone.tapped = AsyncMock()
    provider = phone.state._resolved_provider
    provider.capabilities = {"element_index"}
    provider.get_state.return_value.elements = [{"index": 1, "bounds": "0,0,2,2"}]
    selector = Notes(
        ToolCall("tap_element", {"index": 1}),
        ToolCall("complete", {"success": True, "message": "ok"}),
    )
    result = await AgentOperator(selector=selector).execute(phone, "remember")
    for file in (Path(result.trajectory_path) / "checkpoints").glob("*.json"):
        assert "row seven" not in file.read_text()
    assert (
        "run_notes" in selector.requests[-1].memory and "row seven" in selector.requests[-1].memory
    )


@pytest.mark.anyio
@pytest.mark.parametrize("normalized,configured", [(False, False), (True, False), (True, True)])
async def test_model_coordinates_converted_by_configured_provider(
    config, normalized, configured, monkeypatch
):
    from io import BytesIO

    from mobilerun.core.state.options import StateProviderOptions, screenshot_state_provider
    from mobilerun.utils.images import fit_dimensions_to_max_side
    from PIL import Image

    class Limit:
        def effective_dims(self, width, height):
            return fit_dimensions_to_max_side(width, height, 100)

    phone = Tappable(
        supported_operations=frozenset({"tap"}),
        state_provider_factory=screenshot_state_provider,
        state_options=StateProviderOptions(
            vision=True, normalized=normalized, vision_resize_policy=Limit()
        ),
    )
    phone.tapped = AsyncMock()
    image = BytesIO()
    Image.new("RGB", (400, 200)).save(image, format="PNG")
    phone.screenshot = AsyncMock(return_value=image.getvalue())

    selector = Choices(ToolCall("tap", {"x": 200, "y": 100}))
    if configured:
        from mobilerun_agents.agent.action_selector.loader import SetupSpec

        monkeypatch.setattr(
            "mobilerun_agents.agent.operators.agent.default_setup",
            AsyncMock(return_value=SetupSpec(selector=selector)),
        )
    await AgentOperator(selector=None if configured else selector).execute(
        phone, "tap center", max_steps=1
    )
    req = selector.requests[0]
    assert "tap" in req.by_name and req.state.model_screenshot_width == 100
    assert req.state.use_normalized == normalized
    phone.tapped.assert_awaited_once_with(200, 100)


@pytest.mark.parametrize(
    "profile,flags,enabled",
    [
        (None, [], True),
        ("off", [], False),
        ("off", ["--vision", "on"], True),
        ("on", ["--vision", "off"], False),
        ("off", ["--vision", "only"], True),
    ],
)
def test_run_vision_controls_image_and_coordinate_registry(
    phone, tmp_path, monkeypatch, profile, flags, enabled
):
    from contextlib import asynccontextmanager
    from io import BytesIO

    from click.testing import CliRunner
    from llama_index.core.llms import ImageBlock
    from mobilerun.cli.devices.base.options import DeviceTarget
    from PIL import Image

    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
    from mobilerun_agents.cli.agent.run import run
    from mobilerun_agents.config.loader import ConfigLoader

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-key")
    cfg = MobileConfig(
        profiles={
            "default": LLMProfile(provider="anthropic", model="claude-sonnet-4-5", vision=profile)
        }
    )
    cfg.analytics.enabled = cfg.checkpoints.enabled = False
    ConfigLoader.save(cfg)
    phone._supported_operations = {"tap"}
    provider = phone.state._resolved_provider
    provider.capabilities = {"convert_point"}
    provider.screenshot_matches_input_coords = True
    image = BytesIO()
    Image.new("RGB", (20, 20)).save(image, "PNG")
    phone.screenshot.return_value = image.getvalue()

    @asynccontextmanager
    async def opened(target):
        yield phone

    async def chat(self, messages, **kwargs):
        assert self.vision == (flags[-1] if flags else profile or "on")
        assert any(isinstance(block, ImageBlock) for block in messages[-1].blocks) is enabled
        assert ("tap:" in messages[0].content) is enabled
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content='<function_calls><invoke name="complete"><parameter name="success">true</parameter><parameter name="message">done</parameter></invoke></function_calls>',
            )
        )

    monkeypatch.setattr(DeviceTarget, "open", opened)
    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    result = CliRunner().invoke(run, ["goal", "serial", *flags])
    assert result.exit_code == 0, result.exception
    assert phone.screenshot.await_count == int(enabled)
    assert ConfigLoader.load().profiles["default"].vision == profile


@pytest.mark.parametrize("entry", ["operator", "cli", "user_selector"])
def test_anthropic_token_budget_sets_screenshot_and_tap_space(config, monkeypatch, entry):
    from contextlib import asynccontextmanager
    from io import BytesIO

    from click.testing import CliRunner
    from llama_index.llms.anthropic import Anthropic
    from mobilerun.cli.devices.base.options import DeviceTarget
    from mobilerun.core.state.options import screenshot_state_provider
    from PIL import Image

    from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
    from mobilerun_agents.cli.agent import agent
    from mobilerun_agents.config.loader import ConfigLoader

    phone = Tappable(
        supported_operations=frozenset({"tap"}),
        state_provider_factory=screenshot_state_provider,
    )
    phone.tapped = AsyncMock()
    image = BytesIO()
    Image.new("RGB", (1440, 1920)).save(image, format="PNG")
    phone.screenshot = AsyncMock(return_value=image.getvalue())
    config.profiles["default"] = LLMProfile(provider="anthropic", model="claude-sonnet-4-5")
    calls = []

    async def chat(self, messages, **kwargs):
        calls.append(messages)
        xml = (
            '<invoke name="tap"><parameter name="x">476</parameter><parameter name="y">635</parameter></invoke>'
            if len(calls) == 1
            else '<invoke name="complete"><parameter name="success">true</parameter><parameter name="message">done</parameter></invoke>'
        )
        return ChatResponse(
            message=ChatMessage(role="assistant", content=f"<function_calls>{xml}</function_calls>")
        )

    monkeypatch.setattr(LlamaIndexLLMSelector, "_achat", chat)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unit-key")
    monkeypatch.setattr(ConfigLoader, "load", lambda *_: config)
    original = phone.state._resolved_provider
    if entry == "cli":

        @asynccontextmanager
        async def opened(target):
            yield phone

        monkeypatch.setattr(DeviceTarget, "open", opened)
        result = CliRunner().invoke(agent, ["run", "tap center", "emulator-5554", "--steps", "2"])
        assert result.exit_code == 0, result.exception
    elif entry == "user_selector":
        llm = Anthropic(model="claude-sonnet-4-5", api_key="unit-key")
        config.profiles["default"].provider = "ollama"
        op = AgentOperator(selector=LlamaIndexLLMSelector(llm))
        assert asyncio.run(op.execute(phone, "tap center")).success
    else:
        assert asyncio.run(AgentOperator().execute(phone, "tap center")).success
    assert phone.state._resolved_provider is original and original.vision_resize_policy is None
    assert "952x1270" in calls[0][-1].content
    with Image.open(calls[0][-1].blocks[-1].resolve_image()) as screenshot:
        assert screenshot.size == (952, 1270)
    phone.tapped.assert_awaited_once_with(720, 960)


@pytest.mark.anyio
async def test_sensitive_tool_input_is_not_replayed(phone, config):
    from mobilerun.macro.player import MacroPlayer

    from mobilerun_agents import Run, tool

    @tool(sensitive_parameters=frozenset({"password"}))
    async def enter_password(run: Run, password: str) -> bool:
        """Enter a password."""
        assert run.macro_recorder is not None
        run.macro_recorder.record_action({"action_type": "input_text", "text": password})
        return await run.device.input_text(password)

    phone._supported_operations = {"input_text"}
    phone.state._resolved_provider.capabilities = {"direct_text_input"}
    selector = Choices(
        ToolCall("enter_password", {"password": "synthetic-marker"}),
        ToolCall("wait"),
    )
    result = await AgentOperator(selector=selector, tools=[enter_password]).execute(
        phone, "enter password", max_steps=2
    )
    player = MacroPlayer(phone)
    macro = player.load_macro_from_file(recorded_macro(result.trajectory_path))
    assert [entry["action_type"] for entry in macro["actions"]] == ["wait"]
    assert "synthetic-marker" not in json.dumps(macro)
    assert "enter_password" in selector.requests[1].memory
    assert await player.replay_macro(macro)
    assert phone.inputs == ["synthetic-marker"]


@pytest.mark.anyio
async def test_sensitive_tool_can_record_an_explicit_vault_reference(phone, config):
    from mobilerun.core.credentials import DictCredentials
    from mobilerun.macro.player import MacroPlayer

    from mobilerun_agents import Run, tool

    vault = DictCredentials({"work": "fixture-only"})
    phone.state._resolved_provider.get_state.return_value.phone_state = {
        "focusedElement": {"index": 1},
        "isEditable": True,
    }
    await vault.bind(phone)

    @tool(sensitive_parameters=frozenset({"confirmation"}))
    async def use_vault(run: Run, confirmation: str) -> None:
        """Type the named credential after confirmation."""
        await vault.type_field(None, "password", "work")
        run.macro_recorder.record_action({"action_type": "type_secret", "name": "work"})

    result = await AgentOperator(
        tools=[use_vault],
        selector=Choices(ToolCall("use_vault", {"confirmation": "fixture-confirm"})),
    ).execute(phone, "use vault", max_steps=1)
    player = MacroPlayer(phone)
    macro = player.load_macro_from_file(recorded_macro(result.trajectory_path))
    [entry] = macro["actions"]
    assert {k: v for k, v in entry.items() if k != "pre_state"} == {
        "action_type": "type_secret",
        "name": "work",
    }
    assert "fixture-" not in json.dumps(macro)
    assert await player.replay_macro(macro)
    assert phone.inputs == ["fixture-only", "fixture-only"]


def test_registry_uses_supplied_ui_and_output_schema(phone):
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import tool_run

    run = tool_run(driver=phone, state_provider=phone.state._resolved_provider)
    entry = build_registry(run, [complete], Answer).by_name["complete"]
    assert (
        entry.metadata.get_parameters_dict()["properties"]["output"] == Answer.model_json_schema()
    )


@pytest.mark.anyio
async def test_operator_adds_custom_tool_without_binding_or_exporting_it(phone, config):
    from mobilerun_agents import Run, tool

    before = phone.actions.list()

    @tool(sensitive_parameters=frozenset({"label"}))
    async def read_label(run: Run, label: str) -> str:
        """Read the given bench label."""
        assert run.device is phone and run.device.actions.list() == before
        assert run.ui.formatted_text == "screen"
        return label

    selector = Choices(
        ToolCall("read_label", {"label": "local-only"}),
        ToolCall("complete", {"message": "done", "success": True}),
    )
    result = await AgentOperator(selector=selector, tools=[read_label]).execute(phone, "read label")
    assert result.success and phone.actions.list() == before
    first = selector.requests[0]
    assert first.registry.has_sensitive and "read_label" in first.by_name
    assert 'read_label({"label": "local-only"})' in selector.requests[1].memory
    assert "1 · read_label ✓" in selector.requests[1].memory
    assert json.loads(recorded_macro(result.trajectory_path).read_text())["actions"] == []


def test_base_does_not_import_agent_workflows():
    import ast
    import inspect

    from mobilerun_agents.agent.operators import base

    imports = ast.walk(ast.parse(inspect.getsource(base)))
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            assert "workflows" not in (node.module or "").split(".")[1:]
        elif isinstance(node, ast.Import):
            assert all("workflows" not in alias.name.split(".")[1:] for alias in node.names)


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["agent", "planning"])
async def test_concrete_operator_starts_its_own_workflow(phone, config, monkeypatch, kind):
    from mobilerun_agents.agent.operators import agent, planning
    from mobilerun_agents.agent.operators._run import RunSpec

    workflow = Mock()
    run = RunSpec(phone, "count items", Answer, 7, 3.5, "run-id", config)
    selector, executor = Selector(), Selector()
    from mobilerun_agents import LlamaIndexPlanner

    planner = LlamaIndexPlanner(MockLLM())
    if kind == "planning":
        monkeypatch.setattr(planning, "PlanningWorkflow", workflow)
        operator = planning.PlanningOperator(planner=planner, executor=executor)
    else:
        monkeypatch.setattr(agent, "AgentWorkflow", workflow)
        operator = agent.AgentOperator(selector=selector)
    assert await operator._start(run) is workflow.return_value.run.return_value
    constructors = {"planning": (planner, executor), "agent": ()}
    workflow.assert_called_once_with(*constructors[kind], timeout=3.5)
    inputs = workflow.return_value.run.call_args.kwargs
    assert inputs["device"] is phone and inputs["goal"] == "count items"
    assert (inputs["output"], inputs["max_steps"], inputs["run_id"]) == (Answer, 7, "run-id")
    if kind == "planning":
        assert "action_selector" not in inputs
        assert inputs["memory_factory"]() is not inputs["memory_factory"]()
    else:
        assert inputs["action_selector"] is selector


@pytest.mark.anyio
@pytest.mark.parametrize("override,expected", [(None, 2), (1, 1)])
async def test_execute_resolves_budget_for_workflow(phone, config, override, expected):

    config.agent.max_steps = 2
    config.tools.wait_duration = 0.01
    selector = Choices(*(ToolCall("wait") for _ in range(expected)))
    result = await AgentOperator(selector=selector).execute(phone, "wait", max_steps=override)
    assert result.steps == expected and len(selector.requests) == expected
    checkpoint = Path(result.trajectory_path) / "checkpoints" / f"{expected:04}.json"
    assert json.loads(checkpoint.read_text())["max_steps"] == expected


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize(
    "success,message", [(True, "Checked battery"), (False, "TikTok requires login to read DMs.")]
)
def test_cli_real_steps_visible_without_transport_or_sensitive_args(
    phone, config, monkeypatch, debug, success, message
):
    from contextlib import asynccontextmanager

    from click.testing import CliRunner
    from mobilerun.cli.app import cli
    from mobilerun.cli.devices.base.options import DeviceTarget

    from mobilerun_agents import Run, tool

    @tool(sensitive_parameters=frozenset({"token"}))
    async def authorize(run: Run, token: str) -> str:
        """Authorize the test session."""
        return token

    @asynccontextmanager
    async def opened(target):
        logging.getLogger("mobilerun").debug("Getting state (attempt 1/7)")
        yield phone

    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(Path.home() / "run output"))
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(Path.home() / "captured images"))
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(Path.home() / "saved macros"))
    monkeypatch.delenv("MOBILERUN_LOG_LEVEL", raising=False)
    monkeypatch.setattr(DeviceTarget, "open", opened)

    class CaptureChoices(Choices):
        async def select(self, run):
            await run.screenshot()
            return await super().select(run)

    selector = CaptureChoices(
        ToolCall("authorize", {"token": "fixture-private"}),
        ToolCall("complete", {"message": message, "success": success}),
    )
    monkeypatch.setattr(
        "mobilerun_agents.cli.agent.run.AgentOperator",
        lambda **kwargs: AgentOperator(**kwargs, tools=[authorize]),
    )
    from mobilerun_agents.agent.action_selector.loader import SetupSpec

    monkeypatch.setattr(
        "mobilerun_agents.cli.agent.run.build_setup",
        AsyncMock(return_value=SetupSpec(selector=selector)),
    )
    out = CliRunner().invoke(
        cli, (["--debug"] if debug else []) + ["agent", "run", "battery", "emulator-5554"]
    )
    mark = "✓" if success else "✗"
    assert out.exit_code == (0 if success else 1), out.output
    assert "1 · authorize *** ✓" in out.output and f"2 · complete {mark}" in out.output
    assert out.output.count(f"{mark} {message}") == 1 and "2 steps" in out.output
    run_id = next(get_agent_settings().runs_dir.iterdir()).name
    assert out.stdout.splitlines()[-1] == f"{mark} {message} (2 steps · ~/run output/{run_id})"
    assert (f"~/saved macros/{run_id}/macro.json" in out.stderr) is debug
    assert (f"images folder: ~/captured images/{run_id}" in out.stderr) is debug
    assert "Getting state" not in out.output and "fixture-private" not in out.output


@pytest.mark.anyio
async def test_planning_roles_control_state_and_typed_completion(phone, config, caplog):
    from mobilerun_agents import LlamaIndexPlanner, PlanningOperator
    from tests.unit.agent.workflows.test_planning import PlannerLLM

    class Executor(Choices):
        async def select(self, run):
            assert "set_plan" not in run.registry.by_name and "wait" in run.registry.by_name
            assert run.state.goal in {"Read rows", "Confirm rows"}
            assert (
                "output"
                not in run.registry.by_name["complete"].metadata.get_parameters_dict()["properties"]
            )
            return await super().select(run)

    planner = PlannerLLM(
        ("set_plan", {"subgoals": [" "]}),
        ("set_plan", {"subgoals": ["Read rows", "Confirm rows"]}),
        ("set_plan", {"subgoals": ["Confirm rows"]}),
        ("complete", {"success": True, "message": "verified", "output": {"count": 2}}),
    )

    async def open_url(url: str) -> None:
        assert url.startswith("https://example.invalid/")

    phone._supported_operations = frozenset({"open_url"})
    phone.open_url = open_url
    done = ToolCall("complete", {"success": True, "message": "subgoal done"})
    executor = Executor(
        ToolCall("open_url", {"url": "https://example.invalid/first"}),
        done,
        ToolCall("open_url", {"url": "https://example.invalid/second"}),
        done,
    )
    caplog.set_level(logging.INFO)
    result = await PlanningOperator(planner=LlamaIndexPlanner(planner), executor=executor).execute(
        phone, "count", output=Answer
    )
    assert result.success and result.output == Answer(count=2)
    assert result.steps == 4
    assert [req.goal for req in executor.requests] == ["Read rows"] * 2 + ["Confirm rows"] * 2
    assert "→ subgoal: Confirm rows" in caplog.text
    folder = Path(result.trajectory_path)
    checkpoints = sorted(folder.glob("checkpoints/*.json"))
    assert [checkpoint.stem for checkpoint in checkpoints] == ["0001", "0002", "0003", "0004"]
    macro = recorded_macro(result.trajectory_path)
    assert list(folder.parent.iterdir()) == [folder]
    assert list(macro.parent.parent.glob("*/macro.json")) == [macro]
    data = json.loads(macro.read_text())
    assert data["description"] == "count"
    assert [action["url"] for action in data["actions"]] == [
        "https://example.invalid/first",
        "https://example.invalid/second",
    ]
    for checkpoint in checkpoints:
        assert "Confirm rows" not in checkpoint.read_text()

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection
from mobilerun.core.devices import Device
from mobilerun.core.extensions import Extension, action
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors import DeviceError, NoFocusedInput
from PIL import Image
from pydantic import BaseModel, field_validator

from tests.unit.device_fixtures import MemoryDevice


class Output(BaseModel):
    count: int


class Bench(Extension):
    default_name = "bench"

    @action
    async def ping(self, text: str) -> str:
        """Report the bench status."""
        return text


def call(name, **kwargs):
    return ToolSelection(tool_id=f"call-{name}", tool_name=name, tool_kwargs=kwargs)


def complete(**kwargs):
    return call("complete", success=True, message="done", **kwargs)


class Phone(MemoryDevice):
    def __init__(self, screens, has_tree=False):
        super().__init__()
        self.screens = iter(screens)
        self.captures = 0
        self.has_tree = has_tree
        self._set_state_provider(
            SimpleNamespace(
                get_state=self.refresh, capabilities=set(), requires_coordinate_tools=False
            )
        )

    async def refresh(self):
        return UIState([{"index": 1}] if self.has_tree else [], "screen", "", {}, 10, 10)

    async def context(self):
        return []

    async def screenshot(self):
        self.captures += 1
        return next(self.screens)


class Selector:
    def __init__(self, choices, on_select):
        self.choices = iter(choices)
        self.on_select = on_select
        self.history = []
        self.repairs = []

    async def select(self, ctx, device, memory, registry):
        from mobilerun_agents.agent.action_selector.base import Selection
        from mobilerun_agents.agent.memory import read_memory

        self.history.append(await read_memory(memory, ctx, registry.redactor))
        state = await ctx.store.get_state()
        self.repairs.append(state.repair)
        self.on_select(state)
        choice = next(self.choices)
        if isinstance(choice, Exception):
            raise choice
        return Selection(calls=[choice])


async def run(
    tmp_path,
    monkeypatch,
    choices,
    screens,
    on_select=lambda state: None,
    on_write=lambda: None,
    max_steps=None,
    selector=None,
    has_tree=False,
    **settings,
):
    from mobilerun_agents.agent.memory.memory import build_executor_memory
    from mobilerun_agents.agent.tools.base import tool
    from mobilerun_agents.agent.tools.helpers.images import ImageStore
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.agent.workflows.agent import AgentWorkflow
    from mobilerun_agents.agent.workflows.resources import Services, run_resources
    from mobilerun_agents.config.user_config import MemoryConfig, ToolsConfig

    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path))
    device = Phone(screens, has_tree)
    await Bench().bind(device)
    if selector is None:
        selector = Selector(choices, on_select)

    @tool
    async def write(text: str) -> str:
        """Write a public value."""
        on_write()
        return text

    @tool(sensitive_parameters=frozenset({"text"}))
    async def protect(text: str) -> str:
        """Identify a protected value."""
        return "protected"

    @tool
    async def vault(phone: Device) -> str:
        """Type a secret without exposing the vault value in the call or result."""
        await phone._input_secret('known "late"')
        return "protected"

    memory = build_executor_memory(MemoryConfig(), device.platform)
    images = ImageStore("test", keep=True)
    registry = build_registry(device, [write, protect, vault], ToolsConfig())
    services = Services(registry=registry, images=images)
    workflow = AgentWorkflow(
        selector, resource_manager=await run_resources(device, memory, services), **settings
    )
    handler = workflow.run(goal="Test", max_steps=max_steps)
    try:
        events = [event async for event in handler.stream_events()]
        result = await handler
        return result, await handler.ctx.store.get_state(), events, device, selector
    finally:
        await memory.aclose()


@pytest.mark.parametrize("delay", [None, 0], ids=["default", "disabled"])
def test_observe_settles_after_action_without_charging_a_step(tmp_path, monkeypatch, delay):
    from mobilerun_agents.agent.events import Finished, Observed
    from mobilerun_agents.agent.workflows import agent as workflow
    from mobilerun_agents.config.user_config import AgentConfig

    loading = UIState([], "Loading", "", {}, 10, 10)
    loaded = UIState([], "Loaded", "", {"packageName": "app"}, 10, 10)
    screens = iter([loading, loaded])
    calls = []

    async def refresh(self):
        calls.append("refresh")
        return next(screens)

    async def sleep(seconds):
        calls.append(("sleep", seconds))

    monkeypatch.setattr(Phone, "refresh", refresh)
    monkeypatch.setattr(workflow, "sleep", sleep)
    config = AgentConfig() if delay is None else AgentConfig(settle_delay=delay)
    result, state, events, phone, _ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [call("write", text="open"), complete()],
            [b"A", b"B"],
            config=config,
        )
    )
    assert result.success
    expected = ["refresh", ("sleep", 0.1), "refresh"] if delay is None else ["refresh", "refresh"]
    assert calls == expected
    assert [(event.step, event.package) for event in events if isinstance(event, Observed)] == [
        (1, None),
        (2, "app"),
    ]
    assert [event.steps for event in events if isinstance(event, Finished)] == [2]
    assert state.current.ui.formatted_text == "Loaded"
    assert phone.captures == 2


@pytest.mark.parametrize("delegated", [False, True])
def test_jev_abstention_stops_without_repair_or_fallback(tmp_path, monkeypatch, delegated):
    from mobilerun_agents.agent.action_selector import DecisionSelector, DelegatingSelector
    from mobilerun_agents.agent.events import Repaired
    from tests.unit.agent.action_selector.test_delegating_selector import Model, Stub

    decision = Model(*[("wait", 0.2)] * 4)
    primary = DecisionSelector(decision, "off", min_confidence=0.7)
    delegate = Stub()
    selector = DelegatingSelector(primary, delegate) if delegated else primary

    async def check():
        result, state, events, _, _ = await run(
            tmp_path, monkeypatch, [], [b"A"], selector=selector
        )
        assert not result.success and result.reason == "abstained"
        assert not state.actions and state.repairs == 0
        assert state.current.screenshot is None
        assert not any(isinstance(event, Repaired) for event in events)
        assert len(decision.requests) == 1 and not delegate.calls
        assert state.repair is None

    asyncio.run(check())


@pytest.mark.parametrize(
    ("explicit", "confidence", "target_confidence", "message"),
    [
        (
            False,
            0.32,
            0.61,
            "Unfinished: the decision model's best choice (wait, confidence 0.32) "
            "was below the cutoff 0.7, so no action was taken.",
        ),
        (
            False,
            0.85,
            0.2,
            "Unfinished: the decision model's best choice (wait, target confidence 0.2) "
            "was below the cutoff 0.7, so no action was taken.",
        ),
        (
            True,
            0.32,
            None,
            "Unfinished: the decision model handed the subgoal back without acting "
            "(it chose 'unfinished').",
        ),
    ],
    ids=["below-cutoff", "target-below-cutoff", "explicit"],
)
def test_jev_abstention_outcome_keeps_raw_decision_and_choices(
    tmp_path, monkeypatch, explicit, confidence, target_confidence, message
):
    from mobilerun_agents.agent.action_selector import DecisionSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision
    from tests.unit.agent.action_selector.test_typesafe_decision_model import Model

    def answer(request):
        index = next(i for i, option in enumerate(request.options) if option.action.name == "wait")
        return Decision(
            option_index=None if explicit else index,
            confidence=confidence,
            target_confidence=target_confidence,
            thought="  The search field is visible.\n",
        )

    result, _, _, _, _ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [],
            [b"A"],
            selector=DecisionSelector(Model(answer), "off", min_confidence=0.7),
        )
    )
    assert result.reason == "abstained" and result.message == message
    details = result.jev_abstention
    assert details is not None
    assert details.choice == ("abstain" if explicit else "wait")
    assert details.confidence == confidence and details.cutoff == 0.7
    assert details.target_confidence == target_confidence
    assert details.thought == "  The search field is visible.\n"
    assert "wait 1" in details.choices and "complete 2" in details.choices
    assert "input_text" not in details.choices


def test_abstention_removes_prior_saved_images_for_newly_protected_value(tmp_path, monkeypatch):
    from mobilerun_agents.agent.action_selector.base import Selection

    marker = "fixture-late-abstention"

    class ProtectOnAbstain(Selector):
        async def select(self, ctx, device, memory, registry):
            state = await ctx.store.get_state()
            if len(state.actions) < 2:
                return await super().select(ctx, device, memory, registry)
            assert sorted(path.read_bytes() for path in tmp_path.rglob("*.png")) == [b"A", b"B"]
            registry.protect(call("protect", text=marker))
            return Selection(abstained=True)

    selector = ProtectOnAbstain(
        [call("write", text=marker), call("write", text="next")], lambda _: None
    )
    result, state, *_ = asyncio.run(
        run(tmp_path, monkeypatch, [], [b"A", b"B", b"C"], selector=selector)
    )

    assert result.reason == "abstained"
    assert result.message == "Unfinished: the selector took no action."
    assert not list(tmp_path.rglob("*.png"))
    assert all(record.before.screenshot is None for record in state.actions)
    assert state.current.screenshot is None


def test_happy_path_keeps_pinned_before_and_renders_call_and_result(tmp_path, monkeypatch):
    from mobilerun_agents.agent.events import Acted, Finished, Observed, Selected

    result, state, events, phone, selector = asyncio.run(
        run(tmp_path, monkeypatch, [call("write", text="hello"), complete()], [b"A", b"B"])
    )
    assert result.success and result.message == "done"
    assert [
        type(event) for event in events if type(event) in {Observed, Selected, Acted, Finished}
    ] == [Observed, Selected, Acted, Observed, Selected, Acted, Finished]
    action = state.actions[0]
    assert action.before.screenshot.path.read_bytes() == b"A"
    assert action.call.tool_kwargs == {"text": "hello"}
    assert action.screen_changed is True
    assert phone.captures == 2
    (message,) = selector.history[-1]
    assert message.role == "user"
    assert 'write({"text": "hello"}) · ok=true: hello' in message.content
    assert action.before.screenshot in message.blocks


def test_persisting_image_does_not_modify_selected_snapshot(tmp_path, monkeypatch):
    selected = []
    _, state, *_ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [complete()],
            [b"A"],
            on_select=lambda s: selected.append(s.current),
        )
    )
    assert selected[0].screenshot.path is None
    assert selected[0].screenshot.image is not None
    assert state.actions[0].before.screenshot.path.read_bytes() == b"A"


def test_record_and_memory_use_the_base_action_name(tmp_path, monkeypatch):
    selected = call("bench__ping", text="ready")
    result, state, _, _, selector = asyncio.run(
        run(tmp_path, monkeypatch, [selected, complete()], [b"A", b"B"])
    )
    assert result.success
    record = state.actions[0]
    assert record.ok
    assert record.call.tool_name == "bench.ping"
    assert record.call.tool_kwargs == {"text": "ready"}
    assert record.call.tool_id == selected.tool_id
    (message,) = selector.history[-1]
    assert 'bench.ping({"text": "ready"}) · ok=true:' in message.content


def test_workflows_do_not_render_prompts():
    from mobilerun_agents.agent import workflows

    root = Path(workflows.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "render_prompt" in path.read_text()
    ]
    assert offenders == []


def test_repair_keeps_same_observation_and_records_only_valid_calls(tmp_path, monkeypatch):
    from mobilerun_agents.agent.events import Repaired

    result, state, events, phone, selector = asyncio.run(
        run(tmp_path, monkeypatch, [call("unknown"), call("write", text=12), complete()], [b"A"])
    )
    assert result.success
    assert [record.call.tool_name for record in state.actions] == ["complete"]
    assert phone.captures == 1
    assert sum(1 for event in events if isinstance(event, Repaired)) == 2
    assert [event.reason for event in events if isinstance(event, Repaired)] == selector.repairs[1:]
    assert selector.repairs[0] is None and state.repair is None
    assert all("Invalid arguments" not in message.content for message in selector.history[-1])


def test_repair_budget_ends_without_saving_an_image(tmp_path, monkeypatch):
    result, state, _, phone, selector = asyncio.run(
        run(tmp_path, monkeypatch, [call("unknown")] * 4, [b"A"])
    )
    assert not result.success and result.reason == "repair_attempts"
    assert not state.actions and not list(tmp_path.rglob("*.png"))
    assert phone.captures == 1 and len(selector.history) == 4


@pytest.mark.parametrize("exhausted", [False, True])
def test_rejected_sensitive_selection_forgets_images_before_retry(tmp_path, monkeypatch, exhausted):
    from mobilerun_agents.config.user_config import AgentConfig

    marker = "FAKE_REPAIR_VALUE"
    cfg = AgentConfig()
    cfg.decision.max_response_repair_attempts = 1
    rejected = call("protect", text=marker, unexpected=1)
    choices = [call("write", text=marker if exhausted else "ordinary"), rejected]
    choices.append(rejected if exhausted else complete())
    present = []
    result, state, *_ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            choices,
            [b"A", b"B"],
            config=cfg,
            on_select=lambda _: present.append(bool(list(tmp_path.rglob("*.png")))),
        )
    )
    assert result.reason == ("repair_attempts" if exhausted else "complete")
    assert present == [False, True, False]
    assert state.actions[0].before.screenshot is None
    assert not list(tmp_path.rglob("*.png"))


def test_selector_parse_error_becomes_a_repair(tmp_path, monkeypatch):
    from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
    from mobilerun_agents.agent.events import Repaired

    result, _, events, phone, selector = asyncio.run(
        run(tmp_path, monkeypatch, [InvalidResponse("Malformed model reply"), complete()], [b"A"])
    )
    assert result.success and phone.captures == 1
    assert any(isinstance(event, Repaired) for event in events)
    assert selector.repairs == [None, "Malformed model reply"]


@pytest.mark.parametrize("first_hit", [1, 2, 3, None])
def test_late_protection_discards_affected_files_and_clears_references(
    tmp_path, monkeypatch, first_hit
):
    value = 'known "late"'
    choices = [
        call("write", text=value if first_hit == 1 else "other"),
        call("write", text=value if first_hit == 2 else "more"),
        call("protect", text=value) if first_hit else call("vault"),
        complete(),
    ]

    def inspect(state):
        assert state.current.screenshot.path is None
        assert state.current.screenshot.image is not None
        if len(state.actions) == 2:
            assert sorted(path.read_bytes() for path in tmp_path.rglob("*.png")) == [b"A", b"B"]

    result, state, _, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, choices, [b"A", b"B", b"C", b"D"], inspect)
    )
    assert result.success and phone.captures == 4
    assert (
        sorted(path.read_bytes() for path in tmp_path.rglob("*.png"))
        == {
            1: [],
            2: [b"A"],
            3: [b"A", b"B"],
            None: [b"A"],
        }[first_hit]
    )
    assert [bool(record.before.screenshot) for record in state.actions] == [
        first_hit != 1,
        first_hit == 3,
        False,
        False,
    ]
    assert [record.screen_changed for record in state.actions] == [True, True, True, None]


def test_output_schema_is_validated_before_completing(tmp_path, monkeypatch):
    result, state, *_ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [complete(output={"count": "1"}), complete(output={"count": 2})],
            [b"A", b"A"],
            output=Output,
        )
    )
    assert result.success and result.output == {"count": 2}
    assert [record.ok for record in state.actions] == [False, True]


def test_invalid_outputs_stop_at_the_validation_budget(tmp_path, monkeypatch):
    invalid = complete(output={"count": "FAKE_INVALID_OUTPUT_SECRET"})
    result, state, *_ = asyncio.run(
        run(tmp_path, monkeypatch, [invalid, invalid], [b"A", b"A"], output=Output)
    )
    assert not result.success and result.reason == "output_attempts"
    assert result.message == "count: Input should be a valid integer"
    assert [record.ok for record in state.actions] == [False, False]


def test_completion_without_a_message_records_empty_feedback(tmp_path, monkeypatch):
    result, state, *_ = asyncio.run(
        run(tmp_path, monkeypatch, [call("complete", success=True, message=None)], [b"A"])
    )
    assert result.success and result.message is None
    assert state.actions[0].summary == ""


def test_unchanged_screenshot_is_compared_from_next_observation(tmp_path, monkeypatch):
    _, state, _, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, [call("write", text="hello"), complete()], [b"A", b"A"])
    )
    assert state.actions[0].screen_changed is False
    assert state.actions[1].screen_changed is None
    assert phone.captures == 2


def png(color):
    stream = BytesIO()
    Image.new("RGB", (10, 10), color).save(stream, "PNG")
    return stream.getvalue()


def chat_selector(vision):
    from mobilerun_agents.agent.action_selector.llm.base import LLMSelector

    class Chat(LLMSelector):
        def __init__(self):
            super().__init__(vision)
            self.requests = []
            self.captures_at_entry = []
            self.replies = iter(
                [
                    '<invoke name="write"><parameter name="text">hello</parameter></invoke>',
                    '<invoke name="complete"><parameter name="success">true</parameter>'
                    '<parameter name="message">done</parameter></invoke>',
                ]
            )

        async def select(self, ctx, device, memory, registry):
            self.captures_at_entry.append(device.captures)
            return await super().select(ctx, device, memory, registry)

        async def _achat(self, messages, **kwargs):
            self.requests.append(messages)
            reply = next(self.replies)
            if reply is None:
                return None
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=f"<function_calls>{reply}</function_calls>",
                )
            )

    return Chat()


def test_empty_llm_response_repairs_on_same_observation(tmp_path, monkeypatch):
    from mobilerun_agents.agent.events import Repaired

    selector = chat_selector("off")
    selector.replies = iter(
        [
            None,
            '<invoke name="complete"><parameter name="success">true</parameter>'
            '<parameter name="message">done</parameter></invoke>',
        ]
    )
    result, state, events, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, [], [b"A"], selector=selector)
    )
    reason = "No tool selected; choose an available tool."
    assert result.success and phone.captures == 1 and len(selector.requests) == 2
    assert [event.reason for event in events if isinstance(event, Repaired)] == [reason]
    assert state.repair is None
    assert [message.role for message in selector.requests[1]] == ["system", "user", "user"]
    assert reason not in selector.requests[1][-2].content
    assert f"<repair>{reason}</repair>" in selector.requests[1][-1].content


@pytest.mark.parametrize("vision,captures", [("off", 0), ("on", 2), ("only", 2)])
def test_tree_screenshot_is_only_fetched_by_vision_selector(
    tmp_path, monkeypatch, vision, captures
):
    images = [png("red"), png("blue")]
    selector = chat_selector(vision)
    result, state, _, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, [], images, selector=selector, has_tree=True)
    )
    assert result.success
    assert phone.captures == captures
    assert selector.captures_at_entry == ([0, 1] if captures else [0, 0])
    for record, messages, image in zip(state.actions, selector.requests, images, strict=True):
        assert any(isinstance(block, ImageBlock) for block in messages[-1].blocks) == bool(captures)
        if captures:
            assert record.before.screenshot.path.read_bytes() == image
        else:
            assert record.before.screenshot is None


@pytest.mark.parametrize(
    "second,changed,line",
    [
        (b"B", True, "Previous action changed the screen"),
        (b"A", False, "Previous action did not change the screen"),
    ],
)
def test_screenshot_only_change_is_live_model_context_not_stored_history(
    tmp_path, monkeypatch, second, changed, line
):
    selector = chat_selector("off")
    result, state, _, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, [], [b"A", second], selector=selector)
    )
    assert result.success and state.actions[0].screen_changed is changed
    assert phone.captures == 2 and selector.captures_at_entry == [1, 2]
    assert not selector.requests[0][-1].content.startswith("Previous action")
    assert selector.requests[1][-1].content.splitlines()[0] == line
    assert "<last_step>" in selector.requests[1][-2].content
    assert "Screen changed:" in selector.requests[1][-2].content


@pytest.mark.parametrize(
    "error,expected",
    [
        (RuntimeError("FAKE_UNKNOWN_SDK_SECRET"), "RuntimeError"),
        (DeviceError("FAKE_UNKNOWN_DEVICE_SECRET"), "DeviceError"),
        (NoFocusedInput("Focus an editable field"), "NoFocusedInput: Focus an editable field"),
    ],
)
def test_run_tool_feedback_only_exposes_value_safe_messages(tmp_path, monkeypatch, error, expected):
    def fail():
        raise error

    _, state, _, _, selector = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [call("write", text="hello"), complete()],
            [b"A", b"B"],
            on_write=fail,
        )
    )
    assert not state.actions[0].ok
    (message,) = selector.history[-1]
    assert f"ok=false: {expected}\n" in message.content


def test_untracked_tool_exception_text_is_not_in_public_artifacts(tmp_path, monkeypatch, caplog):
    from mobilerun_agents import AgentOperator, tool
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    marker = "FAKE_UNKNOWN_SDK_SECRET"

    @tool
    async def lookup() -> str:
        """Simulate a failing SDK request."""
        raise RuntimeError(f"https://example.invalid/?api_key={marker}")

    cfg = MobileConfig()
    cfg.checkpoints.enabled = True
    cfg.analytics.enabled = cfg.tracing.enabled = False
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    caplog.set_level(logging.DEBUG, logger="mobilerun_agents.agent.logs.console")
    selector = Selector([call("lookup"), complete()], lambda state: None)
    with ConfigLoader.override(cfg):
        result = asyncio.run(AgentOperator(selector, [lookup]).execute(Phone([b"A", b"B"]), "fake"))

    assert result.success
    checkpoint = (Path(result.trajectory_path) / "context.json").read_text()
    assert "RuntimeError" in caplog.text and "RuntimeError" in checkpoint
    assert marker not in caplog.text and marker not in checkpoint


def test_rejected_output_is_not_echoed_in_public_answer(tmp_path, monkeypatch):
    from mobilerun_agents import AgentOperator
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    cfg = MobileConfig()
    cfg.agent.max_output_validation_attempts = 1
    cfg.checkpoints.enabled = cfg.analytics.enabled = cfg.tracing.enabled = False
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path / "images"))
    selector = Selector(
        [complete(output={"count": "FAKE_INVALID_OUTPUT_SECRET"})], lambda state: None
    )
    with ConfigLoader.override(cfg):
        result = asyncio.run(AgentOperator(selector).execute(Phone([b"A"]), "fake", output=Output))

    assert not result.success and result.answer == "count: Input should be a valid integer"


def test_rejected_output_mapping_key_is_hidden_in_feedback_and_artifacts(
    tmp_path, monkeypatch, caplog
):
    from mobilerun_agents import AgentOperator
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig

    class Counts(BaseModel):
        counts: dict[str, int]

    class TypedOutput(BaseModel):
        results: list[Counts]

    marker = "FAKE_MAPPING_KEY_SECRET"
    cfg = MobileConfig()
    cfg.checkpoints.enabled = True
    cfg.agent.max_output_validation_attempts = 2
    cfg.analytics.enabled = cfg.tracing.enabled = False
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path / "images"))
    caplog.set_level(logging.DEBUG, logger="mobilerun_agents.agent.logs.console")
    rejected = complete(output={"results": [{"counts": {marker: "bad"}}]})
    selector = Selector([rejected, rejected], lambda _: None)
    with ConfigLoader.override(cfg):
        result = asyncio.run(
            AgentOperator(selector).execute(Phone([b"A", b"B"]), "fake", output=TypedOutput)
        )

    assert not result.success
    assert result.answer == "results.0.counts.*: Input should be a valid integer"
    history = [message.model_dump_json() for message in selector.history[-1]]
    assert marker not in "\n".join(history)
    assert marker not in caplog.text
    assert marker not in (Path(result.trajectory_path) / "context.json").read_text()


def test_rejected_custom_validator_exception_text_is_not_echoed(tmp_path, monkeypatch):
    class CustomOutput(BaseModel):
        count: str

        @field_validator("count")
        @classmethod
        def reject(cls, value):
            raise ValueError(f"Cannot accept {value}")

    selected = complete(output={"count": "FAKE_CUSTOM_VALIDATOR_SECRET"})
    result, _, _, _, selector = asyncio.run(
        run(tmp_path, monkeypatch, [selected, selected], [b"A", b"B"], output=CustomOutput)
    )
    assert not result.success
    (message,) = selector.history[-1]
    assert "ok=false: count: Value error\n" in message.content


def test_decision_selector_captures_tree_and_reports_change_as_live_context(tmp_path, monkeypatch):
    from mobilerun_agents.agent.action_selector.decision_model.base import (
        Decision,
        DecisionSelector,
    )

    class Model:
        max_options = None
        locates_points = False

        def __init__(self):
            self.requests = []

        async def decide(self, request):
            self.requests.append(request)
            option = next(
                i
                for i, option in enumerate(request.options)
                if option.action.name == ("wait" if len(self.requests) == 1 else "complete")
                and option.action.parameters.get("success", True)
            )
            return Decision(option_index=option)

    model = Model()
    images = [b"A", b"B"]
    result, state, _, phone, _ = asyncio.run(
        run(tmp_path, monkeypatch, [], images, selector=DecisionSelector(model), has_tree=True)
    )
    assert result.success and phone.captures == 2
    assert [request.screenshot for request in model.requests] == images
    assert [record.before.screenshot.path.read_bytes() for record in state.actions] == images
    assert model.requests[0].previous_action is None
    assert model.requests[1].previous_action == "Previous action did not change the screen"
    assert "Screen changed: false" in model.requests[1].memory


@pytest.mark.anyio
async def test_reused_workflow_passes_the_current_registry():
    from mobilerun_agents.agent.action_selector.base import ActionSelector, Selection
    from mobilerun_agents.agent.tools.base import tool
    from mobilerun_agents.agent.tools.helpers.images import ImageStore
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.agent.workflows.agent import AgentWorkflow
    from mobilerun_agents.agent.workflows.resources import Services, run_resources
    from mobilerun_agents.config.user_config import ToolsConfig

    class Choose(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            name = "second" if "second" in registry.by_name else "first"
            return Selection(calls=[call(name)])

    @tool
    async def first() -> str:
        """Report first."""
        return "first"

    @tool
    async def second() -> str:
        """Report second."""
        return "second"

    phone = Phone([b"A"] * 4)
    images = ImageStore("reuse", keep=False)
    first_registry = build_registry(phone, [first], ToolsConfig())
    second_registry = build_registry(phone, [second], ToolsConfig())
    manager = await run_resources(phone, Memory.from_defaults(), Services(first_registry, images))
    workflow = AgentWorkflow(Choose(), resource_manager=manager)
    await workflow.run(goal="first", max_steps=1)
    second_resources = await run_resources(
        phone, Memory.from_defaults(), Services(second_registry, images)
    )
    for name, resource in second_resources.get_all().items():
        await manager.set(name, resource)
    handler = workflow.run(goal="second", max_steps=1)
    await handler
    state = await handler.ctx.store.get_state()
    assert [record.call.tool_name for record in state.actions] == ["second"]


def test_image_is_not_persisted_during_dispatch_and_terminal_guard_does_not_save(
    tmp_path, monkeypatch
):
    from mobilerun_agents.agent.events import Observed

    files_during_write = []
    result, state, events, phone, _ = asyncio.run(
        run(
            tmp_path,
            monkeypatch,
            [call("write", text="hello")],
            [b"A", b"B"],
            on_write=lambda: files_during_write.append(list(tmp_path.rglob("*.png"))),
            max_steps=1,
        )
    )
    assert files_during_write == [[]]
    assert result.reason == "max_steps"
    assert result.message is None
    assert state.actions[0].screen_changed is True
    assert state.current.screenshot is None
    assert phone.captures == 2
    assert [path.read_bytes() for path in tmp_path.rglob("*.png")] == [b"A"]
    assert [event.step for event in events if isinstance(event, Observed)] == [1]

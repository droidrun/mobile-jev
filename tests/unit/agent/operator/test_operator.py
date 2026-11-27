import asyncio
import json
import logging
from datetime import date
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro.player import MacroPlayer
from mobilerun.macro.settings import get_macro_settings
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    field_serializer,
    field_validator,
)
from workflows import Context

from mobilerun_agents import (
    AgentOperator,
    AgentWorkflow,
    PlanningOperator,
    PlanningWorkflow,
    Selection,
    tool,
)
from mobilerun_agents.agent.action_selector import ActionSelector
from mobilerun_agents.agent.checkpoints.macro_export import macro_data
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory.memory import _RunMemory
from mobilerun_agents.agent.operators import consumers
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import MobileConfig
from tests.unit.device_fixtures import MemoryDevice


class Answer(BaseModel):
    count: int


class Phone(MemoryDevice):
    def __init__(self):
        super().__init__(supported_operations=frozenset({"input_text"}))
        self._set_state_provider(SimpleNamespace(get_state=self.refresh, capabilities=set()))

    async def refresh(self):
        return UIState([], "", "", {}, 0, 0)

    async def context(self):
        return []

    async def screenshot(self):
        return b"fake screenshot"


class Script(ActionSelector):
    def __init__(self, *calls):
        self.calls = iter(calls)

    async def select(self, ctx, device, memory, registry):
        name, args = next(self.calls)
        return Selection(calls=[ToolSelection(tool_id=name, tool_name=name, tool_kwargs=args)])


class ScriptPlanner(Planner):
    def __init__(self):
        self.turn = 0

    async def plan(self, ctx, memory):
        self.turn += 1
        if self.turn == 1:
            return Plan(("enter text",))
        return Finish(success=True, message="done", output={"count": 1})


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    cfg = MobileConfig()
    cfg.analytics.enabled = False
    cfg.tracing.enabled = False
    return cfg


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_execute_maps_result_and_checkpoints_agent_actions(config, planning):
    selector = Script(
        ("input_text", {"text": "sample text"}),
        (
            "complete",
            {"success": True, "message": "done", **({} if planning else {"output": {"count": 1}})},
        ),
    )
    operator = PlanningOperator(ScriptPlanner(), selector) if planning else AgentOperator(selector)
    phone = Phone()
    with ConfigLoader.override(config):
        result = await operator.execute(phone, "type some sample text", output=Answer)
    assert result.success and result.answer == "done"
    assert result.steps == (4 if planning else 2) and result.output == Answer(count=1)
    assert phone.inputs == ["sample text"]
    checkpoint = Path(result.trajectory_path) / "context.json"
    workflow = PlanningWorkflow(ScriptPlanner(), selector) if planning else AgentWorkflow(selector)
    state = await Context.from_dict(workflow, json.loads(checkpoint.read_text())).store.get_state()
    assert [action.step for action in state.actions] == [1, 2]
    screenshot = state.actions[0].before.screenshot
    assert screenshot.path.read_bytes() == b"fake screenshot" and screenshot.image is None
    macro_path = Path(get_macro_settings().macro_directory) / checkpoint.parent.name / "macro.json"
    replay = MemoryDevice()
    player = MacroPlayer(replay)
    macro = player.load_macro_from_file(macro_path)
    assert macro == macro_data(state.goal, state.actions, Redactor())
    assert await player.replay_macro(macro)
    assert replay.inputs == ["sample text"]


@pytest.mark.anyio
async def test_typed_planning_uses_untyped_decision_executor_completion(config):
    from mobilerun_agents.agent.action_selector import DecisionSelector
    from mobilerun_agents.agent.action_selector.decision_model.base import Decision

    class Model:
        max_options = None
        locates_points = False

        async def decide(self, request):
            schema = request.registry["complete"].metadata.get_parameters_dict()
            assert "output" not in schema["properties"]
            index = next(
                i
                for i, option in enumerate(request.options)
                if option.action.name == "complete" and option.action.parameters["success"]
            )
            return Decision(option_index=index)

    class TypedPlanner(ScriptPlanner):
        def bind(self, output_schema, repair, redact):
            assert output_schema == Answer.model_json_schema()
            return self

    class TypedAgent(Script):
        async def select(self, ctx, device, memory, registry):
            schema = registry.by_name["complete"].metadata.get_parameters_dict()
            assert "output" in schema["properties"]
            return await super().select(ctx, device, memory, registry)

    config.checkpoints.enabled = False
    planner = PlanningOperator(TypedPlanner(), DecisionSelector(Model(), vision="off"))
    agent = AgentOperator(
        TypedAgent(("complete", {"success": True, "message": "done", "output": {"count": 2}}))
    )
    with ConfigLoader.override(config):
        result = await planner.execute(Phone(), "read count", output=Answer)
        agent_result = await agent.execute(Phone(), "read count", output=Answer)
    assert result.success and result.steps == 3 and result.output == Answer(count=1)
    assert agent_result.success and agent_result.output == Answer(count=2)


@pytest.mark.anyio
async def test_sensitive_params_remain_raw_in_state_but_not_outputs(config, caplog):
    captured = []

    @tool(sensitive_parameters=frozenset({"text"}))
    async def record_sensitive(text: str) -> str:
        """Remember a sensitive value without publishing it."""
        captured.append(text)
        return f"received {text}"

    selector = Script(
        ("record_sensitive", {"text": "fixture-sensitive-text"}),
        ("complete", {"success": True, "message": "received fixture-sensitive-text"}),
    )
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [record_sensitive]).execute(Phone(), "record it")
    assert captured == ["fixture-sensitive-text"]
    assert result.answer == "received ***"
    assert "fixture-sensitive-text" not in caplog.text
    assert (
        "fixture-sensitive-text" not in (Path(result.trajectory_path) / "context.json").read_text()
    )


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["mailbox", "sms"])
async def test_sensitive_extension_result_is_raw_for_model_only(config, caplog, kind):
    from llama_index.core.llms import ChatMessage, ChatResponse
    from mobilerun.core.extensions import action
    from mobilerun.core.mailbox import Mailbox
    from mobilerun.core.phone_number import PhoneNumber
    from mobilerun_sdk.types.numbers.message_list_response import Item

    from mobilerun_agents.agent.action_selector.llm.base import LLMSelector

    code = "314159"
    secret = code if kind == "mailbox" else f"Year 2026. Your fixture SMS code is {code}."

    class Mail(Mailbox):
        address = None

        @action
        async def wait_for_code(self) -> str:
            """Read the fixture verification code."""
            return secret

    class Number(PhoneNumber):
        id = "fake"
        phone_number = device_id = None

        async def wait_for_code(self):
            raise AssertionError("only messages should be read")

        async def send(self, to, text):
            raise AssertionError("no SMS should be sent")

        @action
        async def messages(self) -> list[Item]:
            """Read the fixture SMS."""
            return [
                Item(
                    id="fixture-message",
                    body=secret,
                    createdAt="2026-01-01T00:00:00Z",
                    occurredAt="2026-01-01T00:00:00Z",
                    direction="inbound",
                    status="received",
                )
            ]

    class Model(LLMSelector):
        requests = []

        async def _achat(self, messages, **kwargs):
            self.requests.append(messages)
            match len(self.requests):
                case 1:
                    text = f'<invoke name="{tool_name}"></invoke>'
                case 2:
                    text = f'<invoke name="input_text"><parameter name="text">{code}</parameter></invoke>'
                case _:
                    text = (
                        '<invoke name="complete"><parameter name="success">true</parameter>'
                        f'<parameter name="message">Code: {code}</parameter></invoke>'
                    )
            return ChatResponse(
                message=ChatMessage(
                    role="assistant", content=f"<function_calls>{text}</function_calls>"
                )
            )

    phone = Phone()
    phone.state._resolved_provider.requires_coordinate_tools = False
    extension, tool_name = (
        (Mail(), "email__wait_for_code") if kind == "mailbox" else (Number(), "sms__messages")
    )
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    model = Model(vision="off")
    await extension.bind(phone)
    try:
        with ConfigLoader.override(config):
            result = await AgentOperator(model).execute(phone, "Read a message")
    finally:
        await extension.unbind()
    assert result.success and len(model.requests) == 3 and phone.inputs == [code]
    assert result.answer == "Code: ***"
    assert any(secret in (message.content or "") for message in model.requests[1])
    folder = Path(result.trajectory_path)
    checkpoint = (folder / "context.json").read_text()
    memory_dump = "\n".join(path.read_text() for path in sorted((folder / "memory").glob("*.txt")))
    for public in (checkpoint, memory_dump, caplog.text):
        assert secret not in public and code not in public and "***" in public


@pytest.mark.anyio
async def test_debug_operator_writes_masked_memory_file(config, caplog):
    from mobilerun_agents.agent.logs.memory import MemoryDetail

    class MemorySelector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            registry.redactor.add("fixture-memory-secret")
            ctx.write_event_to_stream(
                MemoryDetail(
                    section="memory",
                    text="fixture-memory-secret",
                    blocks={},
                    last_steps=[],
                    counts="1 msg",
                )
            )
            return Selection(
                calls=[
                    ToolSelection(
                        tool_id="1",
                        tool_name="complete",
                        tool_kwargs={"success": True, "message": "done"},
                    )
                ]
            )

    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    with ConfigLoader.override(config):
        result = await AgentOperator(MemorySelector()).execute(Phone(), "goal")
    saved = Path(result.trajectory_path) / "memory/0001.txt"
    assert saved.read_text() == "***\n"
    assert "fixture-memory-secret" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_execution_result_masks_sensitive_values_in_typed_output(config, planning):
    marker = "fixture-sensitive-output"

    class TypedOutput(BaseModel):
        details: dict[str, list[str]]

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> str:
        """Receive a sensitive value."""
        return "done"

    message = f"Received {marker}"
    raw_output = {"details": {"text": [marker, "ordinary"]}}

    class OutputPlanner(ScriptPlanner):
        async def plan(self, ctx, memory):
            if (await ctx.store.get_state()).results:
                return Finish(success=True, message=message, output=raw_output)
            return Plan(("receive the value",))

    completion = {"success": True, "message": message}
    if not planning:
        completion["output"] = raw_output
    selector = Script(("protect", {"value": marker}), ("complete", completion))
    operator = (
        PlanningOperator(OutputPlanner(), selector, [protect])
        if planning
        else AgentOperator(selector, [protect])
    )
    with ConfigLoader.override(config):
        result = await operator.execute(Phone(), "receive the value", output=TypedOutput)
    assert result.success and result.answer == "Received ***"
    assert result.output == TypedOutput(details={"text": ["***", "ordinary"]})


@pytest.mark.anyio
async def test_execution_result_omits_output_if_mask_breaks_its_schema(config):
    marker = "fixture-sensitive-constrained-output"

    class ConstrainedOutput(BaseModel):
        text: str = Field(min_length=5)

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> str:
        """Receive a sensitive value."""
        return "done"

    selector = Script(
        ("protect", {"value": marker}),
        ("complete", {"success": True, "message": "done", "output": {"text": marker}}),
    )
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [protect]).execute(
            Phone(), "receive the value", output=ConstrainedOutput
        )
    assert result.success and result.output is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "success, expected",
    [(True, "Goal achieved."), (False, "Goal cannot be achieved with the available actions.")],
)
async def test_absent_completion_message_only_gets_a_display_default(
    config, caplog, success, expected
):
    selector = Script(("complete", {"success": success, "message": None}))
    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    with ConfigLoader.override(config):
        result = await AgentOperator(selector).execute(Phone(), "do the task")
    assert result.answer == expected and result.success is success
    assert any(expected in message for message in caplog.messages)
    source = (Path(result.trajectory_path) / "context.json").read_text()
    restored = Context.from_dict(AgentWorkflow(selector), json.loads(source))
    assert (await restored.store.get_state()).outcome.message is None
    assert expected not in source


@pytest.mark.anyio
async def test_subgoal_display_default_does_not_replace_raw_message(config, caplog):
    replanning = asyncio.Event()
    may_finish = asyncio.Event()
    reported = asyncio.Event()

    class PausedPlanner(ScriptPlanner):
        async def plan(self, ctx, memory):
            if (await ctx.store.get_state()).results:
                replanning.set()
                await may_finish.wait()
            return await super().plan(ctx, memory)

    class ProgressLog(logging.Handler):
        def emit(self, record):
            if record.getMessage() == "✓ subgoal: Goal achieved.":
                reported.set()

    planner = PausedPlanner()
    selector = Script(("complete", {"success": True, "message": None}))
    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    logger = logging.getLogger("mobilerun_agents.agent.logs.console")
    progress_log = ProgressLog()
    logger.addHandler(progress_log)
    try:
        with ConfigLoader.override(config):
            task = asyncio.create_task(
                PlanningOperator(planner, selector).execute(Phone(), "do the task")
            )
            try:
                await asyncio.wait_for(replanning.wait(), timeout=5)
                await asyncio.wait_for(reported.wait(), timeout=5)
            finally:
                may_finish.set()
                result = await task
    finally:
        logger.removeHandler(progress_log)
    assert result.answer == "done"
    assert "✓ subgoal: Goal achieved." in caplog.messages
    source = (Path(result.trajectory_path) / "context.json").read_text()
    restored = Context.from_dict(PlanningWorkflow(planner, selector), json.loads(source))
    state = await restored.store.get_state()
    assert state.results[0].message is None
    assert "Goal achieved." not in source


@pytest.mark.anyio
async def test_vault_value_stays_out_of_checkpoint_console_and_macro(config, caplog):
    class EditablePhone(Phone):
        async def refresh(self):
            return UIState([], "", "", {"focusedElement": {"text": ""}, "isEditable": True}, 0, 0)

    marker = "vault-fixture-credential"
    phone = EditablePhone()
    credentials = await DictCredentials({"app": {"sample": {"password": marker}}}).bind(phone)
    selector = Script(
        ("vault__type_field", {"package": "app", "name": "sample", "field": "password"}),
        ("complete", {"success": True, "message": f"Entered {marker}"}),
    )
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    try:
        with ConfigLoader.override(config):
            result = await AgentOperator(selector).execute(phone, "enter the saved password")
    finally:
        await credentials.unbind()
    assert phone.inputs == [marker]
    assert result.answer == "Entered ***" and marker not in caplog.text
    folder = Path(result.trajectory_path)
    assert marker not in (folder / "context.json").read_text()
    macro_path = Path(get_macro_settings().macro_directory) / folder.name / "macro.json"
    macro = MacroPlayer().load_macro_from_file(macro_path)
    assert marker not in macro_path.read_text()
    assert macro["actions"][0]["name"] == "vault.type_field"


@pytest.mark.anyio
@pytest.mark.parametrize("exit_kind", ["success", "error", "cancel"])
async def test_images_close_on_success_error_and_cancellation(config, monkeypatch, exit_kind):
    seen = asyncio.Event()
    closed = []
    original = consumers.ImageStore.close

    def close(store):
        closed.append(store.folder)
        original(store)

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            seen.set()
            if exit_kind == "error":
                raise RuntimeError("fake model failure")
            if exit_kind == "cancel":
                await asyncio.Event().wait()
            return Selection(
                calls=[
                    ToolSelection(
                        tool_id="1",
                        tool_name="complete",
                        tool_kwargs={"success": True, "message": "done"},
                    )
                ]
            )

    monkeypatch.setattr(consumers.ImageStore, "close", close)
    config.checkpoints.enabled = False
    with ConfigLoader.override(config):
        run = asyncio.create_task(AgentOperator(Selector()).execute(Phone(), "goal"))
        await asyncio.wait_for(seen.wait(), 5)
        if exit_kind == "cancel":
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        elif exit_kind == "error":
            with pytest.raises(Exception, match="fake model failure"):
                await run
        else:
            assert (await run).success
    assert len(closed) == 1 and not closed[0].exists()


@pytest.mark.anyio
async def test_abnormal_planning_finish_reports_shared_step_count(config, monkeypatch):
    finished = []

    class FailingSelector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            raise RuntimeError("fixture executor failure")

    original_finish = consumers.ConsoleLog.finish

    def capture(console, success, message, steps, folder):
        finished.append(steps)
        return original_finish(console, success, message, steps, folder)

    monkeypatch.setattr(consumers.ConsoleLog, "finish", capture)
    with (
        ConfigLoader.override(config),
        pytest.raises(RuntimeError, match="fixture executor failure"),
    ):
        await PlanningOperator(ScriptPlanner(), FailingSelector()).execute(Phone(), "goal")
    assert finished == [1]


@pytest.mark.anyio
async def test_cancellation_during_second_planner_turn_reports_three_steps(config, caplog):
    second_turn = asyncio.Event()

    class BlockingPlanner(ScriptPlanner):
        async def plan(self, ctx, memory):
            if self.turn == 1:
                second_turn.set()
                await asyncio.Future()
            return await super().plan(ctx, memory)

    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    config.checkpoints.enabled = False
    selector = Script(("complete", {"success": True, "message": "done"}))
    with ConfigLoader.override(config):
        run = asyncio.create_task(
            PlanningOperator(BlockingPlanner(), selector).execute(Phone(), "goal")
        )
        try:
            await asyncio.wait_for(second_turn.wait(), 5)
        finally:
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run

    assert "cancelled (3 steps)" in caplog.text


class Aliased(BaseModel):
    order_id: str = Field(alias="orderId")


class SplitAliases(BaseModel):
    order_id: str = Field(validation_alias="incomingId", serialization_alias="outgoingId")


class Status(str, Enum):
    READY = "ready"


class StrictOutput(BaseModel):
    model_config = ConfigDict(strict=True)
    day: date
    status: Status


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"orderId": "A1"}, Aliased(orderId="A1")),
        ({"incomingId": "A1"}, SplitAliases(incomingId="A1")),
        (
            {"day": "2026-01-02", "status": "ready"},
            StrictOutput(day=date(2026, 1, 2), status=Status.READY),
        ),
        (None, RootModel[None](None)),
    ],
)
async def test_typed_output_preserves_validated_json(config, planning, raw, expected):
    class Planner(ScriptPlanner):
        async def plan(self, ctx, memory):
            return Finish(success=True, message="done", output=raw)

    selector = Script(("complete", {"success": True, "message": "done", "output": raw}))
    operator = PlanningOperator(Planner(), selector) if planning else AgentOperator(selector)
    with ConfigLoader.override(config):
        result = await operator.execute(Phone(), "read order", output=type(expected))

    assert result.success and result.output == expected
    if not planning:
        checkpoint = Path(result.trajectory_path) / "context.json"
        state = await Context.from_dict(
            AgentWorkflow(selector), json.loads(checkpoint.read_text())
        ).store.get_state()
        assert state.outcome.output == raw


@pytest.mark.anyio
async def test_invalid_redacted_output_warns_without_values(config, caplog):
    marker = "fixture-warning-protected"

    class Output(BaseModel):
        text: str = Field(min_length=5)

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> str:
        """Protect a fixture value."""
        return "done"

    selector = Script(
        ("protect", {"value": marker}),
        ("complete", {"success": True, "message": "done", "output": {"text": marker}}),
    )
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [protect]).execute(Phone(), "read", output=Output)

    assert result.success and result.output is None
    warnings = [
        r for r in caplog.records if r.name == consumers.__name__ and r.levelname == "WARNING"
    ]
    assert len(warnings) == 1 and "redact" in warnings[0].getMessage().lower()
    assert marker not in warnings[0].getMessage() and "***" not in warnings[0].getMessage()


@pytest.mark.anyio
@pytest.mark.parametrize("source", ["default", "validator", "safe"])
async def test_public_output_checks_values_after_revalidation(config, caplog, source):
    marker = "fixture-protected-default"

    class DefaultOutput(BaseModel):
        text: str = marker

    class ValidatorOutput(BaseModel):
        text: str

        @field_validator("text")
        @classmethod
        def add_protected_value(cls, value: str) -> str:
            return marker

    class SafeOutput(BaseModel):
        masked: str = Field(validation_alias="incoming", serialization_alias="outgoing")
        nested: StrictOutput
        optional: str | None = None

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> str:
        """Protect a fixture value."""
        return "done"

    output, raw = {
        "default": (DefaultOutput, {}),
        "validator": (ValidatorOutput, {"text": "ordinary"}),
        "safe": (
            SafeOutput,
            {"incoming": marker, "nested": {"day": "2026-01-02", "status": "ready"}},
        ),
    }[source]
    selector = Script(
        ("protect", {"value": marker}),
        ("complete", {"success": True, "message": "done", "output": raw}),
    )
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [protect]).execute(Phone(), "read", output=output)

    assert result.success
    warnings = [r.message for r in caplog.records if r.name == consumers.__name__]
    if source == "safe":
        assert result.output == SafeOutput(
            incoming="***", nested=StrictOutput(day=date(2026, 1, 2), status=Status.READY)
        )
        assert not warnings
    else:
        assert result.output is None
        assert warnings == ["Typed output omitted: redacted output did not match the schema"]


@pytest.mark.anyio
@pytest.mark.parametrize("use_default", [False, True], ids=["masked-input", "protected-default"])
async def test_public_output_omits_unverified_model_with_contextual_serializer(
    config, caplog, use_default
):
    marker = "fixture-contextual-output"

    class ContextualOutput(BaseModel):
        text: str = marker

        @field_serializer("text")
        def serialize_text(self, value: str, info: SerializationInfo) -> str:
            return info.context["prefix"] + value

    class Selector(Script):
        async def select(self, ctx, device, memory, registry):
            registry.redactor.add(marker)
            return await super().select(ctx, device, memory, registry)

    selector = Selector(
        (
            "complete",
            {"success": True, "message": "done", "output": {} if use_default else {"text": marker}},
        )
    )
    with ConfigLoader.override(config):
        result = await AgentOperator(selector).execute(Phone(), "read", output=ContextualOutput)

    assert result.success and result.output is None
    assert [r.message for r in caplog.records if r.name == consumers.__name__] == [
        "Typed output omitted: redacted output did not match the schema"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("failed_save", ["acted", "final"])
async def test_failed_checkpoint_write_refreshes_old_redactions(config, monkeypatch, failed_save):
    marker = "fixture-checkpoint-late-secret"
    config.checkpoints.enabled = True
    original = consumers.save_context
    attempts = []

    @tool
    async def note() -> str:
        """Return a value not yet known to be sensitive."""
        return marker

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> str:
        """Protect a fixture value."""
        return "done"

    async def save(path, ctx, redactor):
        attempts.append(path)
        number = len(attempts)
        if number == 1:
            assert not redactor.revision
        if failed_save == "final" and number == 3:
            # Learn the value after the last step save, before the final save starts.
            saved = await original(path, ctx, redactor)
            redactor.add(marker)
            return saved
        if number == (2 if failed_save == "acted" else 4):
            assert marker in path.read_text() and redactor.revision
            raise OSError("checkpoint write failed")
        return await original(path, ctx, redactor)

    monkeypatch.setattr(consumers, "save_context", save)
    second = ("protect", {"value": marker}) if failed_save == "acted" else ("note", {})
    selector = Script(("note", {}), second, ("complete", {"success": True, "message": "done"}))
    with ConfigLoader.override(config), pytest.raises(OSError, match="checkpoint write failed"):
        await AgentOperator(selector, [note, protect]).execute(Phone(), "read")

    checkpoint: Path = attempts[0]
    assert checkpoint.is_file() and marker not in checkpoint.read_text()


@pytest.mark.anyio
@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancellation"])
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("planning", [False, True])
async def test_run_memory_closes_after_workflow_and_checkpoints(
    config, monkeypatch, caplog, exit_kind, close_fails, planning
):
    started = asyncio.Event()
    operations = []
    used = []
    closed = []
    original_close = _RunMemory.aclose
    original_save = consumers.save_context

    async def close(memory):
        operations.append("close")
        closed.append(memory)
        await original_close(memory)
        if close_fails:
            raise RuntimeError("fixture close failure")

    async def save(path, ctx, redactor):
        saved = await original_save(path, ctx, redactor)
        operations.append("checkpoint")
        return saved

    async def stop(memory):
        used.append(memory)
        started.set()
        try:
            if exit_kind == "failure":
                raise ValueError("fixture workflow failure")
            if exit_kind == "cancellation":
                await asyncio.Future()
        finally:
            operations.append("workflow exited")

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            await stop(memory)
            return Selection(
                calls=[
                    ToolSelection(
                        tool_id="complete",
                        tool_name="complete",
                        tool_kwargs={"success": True, "message": "done"},
                    )
                ]
            )

    class Planner(ScriptPlanner):
        async def plan(self, ctx, memory):
            await stop(memory)
            return Finish(True, "done")

    monkeypatch.setattr(_RunMemory, "aclose", close)
    monkeypatch.setattr(consumers, "save_context", save)
    operator = PlanningOperator(Planner(), Selector()) if planning else AgentOperator(Selector())
    with ConfigLoader.override(config):
        run = asyncio.create_task(operator.execute(Phone(), "goal"))
        await asyncio.wait_for(started.wait(), 5)
        if exit_kind == "cancellation":
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
        elif exit_kind == "failure":
            with pytest.raises(ValueError, match="fixture workflow failure"):
                await run
        else:
            assert (await run).success

    checkpoints = ["checkpoint"] * (1 if planning else 2) if exit_kind == "success" else []
    assert operations == ["workflow exited", *checkpoints, "close"]
    assert len(used) == len(closed) == 1 and used[0] is closed[0]
    if planning:
        assert [block.name for block in used[0].memory_blocks] == config.memory.planner.blocks
    assert ("Unable to close run memory" in caplog.text) == close_fails
    assert "fixture close failure" not in caplog.text

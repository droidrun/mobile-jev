import asyncio
import json
import logging
from datetime import date
from enum import Enum
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import RecordingDevice, StealthDevice
from mobilerun.core.state.ui_state import UIState
from mobilerun.macro import load_macro
from mobilerun.macro.settings import get_macro_settings
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    field_serializer,
)
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun_agents import (
    AgentOperator,
    PlanningOperator,
    Selection,
    tool,
)
from mobilerun_agents.agent.action_selector import ActionSelector
from mobilerun_agents.agent.events import Acted, Finished
from mobilerun_agents.agent.memory import read_memory
from mobilerun_agents.agent.memory.memory import _RunMemory
from mobilerun_agents.agent.operators import consumers
from mobilerun_agents.agent.planner import Finish, Plan, Planner
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.agent.workflows.core.state import Outcome
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import MobileConfig
from tests.unit.device_fixtures import MemoryDevice


class Answer(BaseModel):
    count: int


class Phone(MemoryDevice):
    def __init__(self):
        super().__init__(supported_operations=frozenset({"input_text"}))
        self._set_state_provider(
            SimpleNamespace(
                get_state=self.refresh, get_final_state=self.refresh, capabilities=set()
            )
        )

    async def refresh(self):
        return UIState([], "", "", {"packageName": "fixture.app"}, 0, 0)

    async def context(self):
        return []

    async def screenshot(self, hide_overlay=True):
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


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_configured_notes_are_callable_and_visible_next_turn(planning, enabled):
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            assert ("note" in registry.by_name) is enabled
            messages = await read_memory(memory, ctx)
            if enabled and not (await ctx.store.get_state()).notes:
                return Selection(
                    calls=[
                        ToolSelection(
                            tool_id="save", tool_name="note", tool_kwargs={"text": "Executor fact"}
                        )
                    ]
                )
            if enabled:
                assert any("Executor fact" in message.content for message in messages)
            return Selection(
                calls=[
                    ToolSelection(
                        tool_id="done",
                        tool_name="complete",
                        tool_kwargs={"success": True, "message": "done"},
                    )
                ]
            )

    config = MobileConfig(recording={"enabled": False})
    if enabled:
        config.memory.executor.blocks.append("notes")
        config.memory.planner.blocks.append("notes")
    llm = Replies()
    llm._calls = ([("note", {"text": "Planner fact"})] if enabled else []) + [
        ("set_plan", {"subgoals": ["Look"], "findings": ["Screen fact"]}),
        ("complete", {"success": True, "message": "done"}),
    ]
    operator = (
        PlanningOperator(LlamaIndexPlanner(llm), Selector())
        if planning
        else AgentOperator(Selector())
    )
    with ConfigLoader.override(config):
        result = await operator.execute(Phone(), "Save observations", max_steps=5)
    assert result.success
    if planning:
        assert result.steps == (5 if enabled else 3)
        assert ("note" in [tool.metadata.get_name() for tool in llm._requests[0][1]]) is enabled
        assert any("Screen fact" in msg.content for msg in llm._requests[-1][0])
        if enabled:
            assert any("Planner fact" in msg.content for msg in llm._requests[1][0])
            assert any("Planner fact" in msg.content for msg in llm._requests[-1][0])
            assert any("Executor fact" in msg.content for msg in llm._requests[-1][0])
            assert [tool.metadata.get_name() for tool in llm._requests[-1][1]] == ["complete"]
    else:
        assert result.steps == (2 if enabled else 1)


@pytest.mark.anyio
async def test_planner_can_save_note_without_budget_for_another_subgoal():
    from mobilerun_agents.agent.planner import LlamaIndexPlanner
    from tests.unit.agent.planner.test_llama_index import Replies

    config = MobileConfig(recording={"enabled": False}, memory={"planner": {"blocks": ["notes"]}})
    llm = Replies()
    llm._calls = [
        ("note", {"text": "Battery full"}),
        ("complete", {"success": True, "message": "done"}),
    ]
    with ConfigLoader.override(config):
        result = await PlanningOperator(LlamaIndexPlanner(llm), Script()).execute(
            Phone(), "Check the battery", max_steps=2
        )
    assert result.success and result.steps == 2
    (first, first_tools), (second, second_tools) = llm._requests
    assert [tool.metadata.get_name() for tool in first_tools] == ["note", "complete"]
    assert [tool.metadata.get_name() for tool in second_tools] == ["complete"]
    assert all("Battery full" not in message.content for message in first)
    assert any("Battery full" in message.content for message in second)
    assert "Note saved." in second[-1].content


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.delenv("MOBILERUN_AGENT_RECORDING_ENABLED", raising=False)
    monkeypatch.setenv("MOBILERUN_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_RUN_SETTLE_SECONDS", "0")
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    cfg = MobileConfig()
    cfg.analytics.enabled = False
    cfg.tracing.enabled = False
    return cfg


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_execute_maps_result_and_exports_recorded_actions(config, planning):
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
    folder = Path(result.trajectory_path)
    manifest = json.loads((folder / "run.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["kind"] == "agent"
    assert {path.name for path in folder.iterdir()} == {"agent.json", "run.json", "steps.jsonl"}
    steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
    assert [(step["i"], step["action"], step["by"]) for step in steps] == [
        (1, "input_text", "agent")
    ]
    evidence = json.loads((folder / "agent.json").read_text())
    assert [(row["text"], row["step_ids"]) for row in evidence if row["kind"] == "plan"] == (
        [("enter text", [1])] if planning else []
    )
    assert [(row["text"], row["step_ids"]) for row in evidence if row["kind"] == "tool"] == [
        ("input_text", [1]),
        ("complete", []),
    ]
    macro_path = Path(get_macro_settings().macro_directory) / folder.name / "v1.jsonl"
    assert [(step.i, step.action, step.args["text"]) for step in load_macro(macro_path)] == [
        (1, "input_text", "sample text")
    ]


@pytest.mark.anyio
async def test_thought_links_to_entire_batch_but_tool_entries_keep_exact_links(config):
    class Batch(Script):
        async def select(self, ctx, device, memory, registry):
            if not device._unwrap().inputs:
                return Selection(
                    calls=[
                        ToolSelection(
                            tool_id=str(i), tool_name="input_text", tool_kwargs={"text": value}
                        )
                        for i, value in enumerate(("first", "second"))
                    ],
                    thought="Enter both values",
                )
            return await super().select(ctx, device, memory, registry)

    phone = Phone()
    with ConfigLoader.override(config):
        result = await AgentOperator(
            Batch(("complete", {"success": True, "message": "done"}))
        ).execute(phone, "Type two values")

    assert result.success and phone.inputs == ["first", "second"]
    evidence = json.loads((Path(result.trajectory_path) / "agent.json").read_text())
    assert [(entry["text"], entry["step_ids"]) for entry in evidence] == [
        ("Enter both values", [1, 2]),
        ("input_text", [1]),
        ("input_text", [2]),
        ("complete", []),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "kind", ["stealth-inner", "stealth-outer", "disabled", "config-disabled", "reused"]
)
async def test_caller_recorder_ownership_and_opt_out(config, tmp_path, caplog, kind):
    from mobilerun_agents.agent.action_selector.base import screenshot_bytes

    seen = []

    class VisionScript(Script):
        async def select(self, ctx, device, memory, registry):
            state = await ctx.store.get_state()
            seen.append(await screenshot_bytes(ctx, state.current, device))
            return await super().select(ctx, device, memory, registry)

    phone = Phone()
    recorder = RecordingDevice(
        StealthDevice(phone) if kind == "stealth-inner" else phone,
        runs_dir=tmp_path / "runs",
        capture="full",
    )
    device = StealthDevice(recorder) if kind == "stealth-outer" else recorder
    selector = VisionScript(
        ("input_text", {"text": "fixture"}),
        ("complete", {"success": True, "message": "done"}),
    )
    caplog.set_level(logging.DEBUG, logger="mobilerun_agents.agent.logs.console")
    if kind == "config-disabled":
        config.recording.enabled = False
    with ConfigLoader.override(config):
        result = await AgentOperator(
            selector, record=False if kind == "disabled" else None
        ).execute(device, "type")
    assert result.success and phone.inputs == ["fixture"]
    assert seen == [b"fake screenshot", b"fake screenshot"]
    assert not (tmp_path / "macros").exists()
    if kind == "reused":
        assert recorder.run_dir is not None and result.trajectory_path == str(recorder.run_dir)
        manifest = json.loads((recorder.run_dir / "run.json").read_text())
        step = json.loads((recorder.run_dir / "steps.jsonl").read_text())
        assert manifest["status"] == "running" and step["by"] == "agent"
        assert step["before"]["app"] == step["after"]["app"] == "fixture.app"
        assert {
            p.relative_to(recorder.run_dir).as_posix()
            for p in recorder.run_dir.rglob("*")
            if p.is_file()
        } == {
            "agent.json",
            "run.json",
            "steps.jsonl",
            "screens/1.before.json",
            "screens/1.after.json",
            "screens/1.before.png",
            "screens/1.after.png",
        }
        await recorder.input_text("caller still recording")
        assert recorder.step_index == 2
    else:
        assert result.trajectory_path is None and recorder.step_index == 0
        assert not (tmp_path / "runs").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("exit_kind", ["unsuccessful", "error", "cancel"])
async def test_owned_recording_finalizes_without_macro_on_unsuccessful_exit(
    config, tmp_path, exit_kind
):
    acted = asyncio.Event()

    class Selector(Script):
        async def select(self, ctx, device, memory, registry):
            if phone.inputs:
                acted.set()
                if exit_kind == "error":
                    raise RuntimeError("fixture failure")
                if exit_kind == "cancel":
                    await asyncio.Future()
            return await super().select(ctx, device, memory, registry)

    phone = Phone()
    selector = Selector(
        ("input_text", {"text": "fixture"}),
        ("complete", {"success": False, "message": "not done"}),
    )
    with ConfigLoader.override(config):
        task = asyncio.create_task(AgentOperator(selector).execute(phone, "type"))
        await asyncio.wait_for(acted.wait(), 5)
        if exit_kind == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_kind == "error":
            with pytest.raises(RuntimeError, match="fixture failure"):
                await task
        else:
            assert not (await task).success
    (folder,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((folder / "run.json").read_text())
    assert manifest["status"] == ("cancelled" if exit_kind == "cancel" else "failed")
    assert json.loads((folder / "steps.jsonl").read_text())["ok"] is True
    assert not (tmp_path / "macros").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_operator_memory_blocks_are_available_to_each_agent(config, monkeypatch, planning):
    from mobilerun_agents import StateBlock
    from mobilerun_agents.agent.memory import registry as block_registry
    from mobilerun_agents.agent.workflows.core.state import AgentState, PlanningState

    monkeypatch.setattr(metadata, "entry_points", lambda **kwargs: ())
    block_registry._installed.cache_clear()

    class LocalBlock(StateBlock[AgentState | PlanningState, str]):
        name: str = "local"
        priority: int = 3
        role: str

        async def render(self, state):
            return f"{self.role}: {state.goal}"

    class Selector(Script):
        async def select(self, ctx, device, memory, registry):
            (message,) = await read_memory(memory, ctx)
            goal = next(goals)
            assert f"executor: {goal}" in message.content
            child_memories.append(memory)
            return await super().select(ctx, device, memory, registry)

    class LocalPlanner(ScriptPlanner):
        async def plan(self, ctx, memory):
            (message,) = await read_memory(memory, ctx)
            assert "planner: read count" in message.content
            self.turn += 1
            if self.turn <= 2:
                return Plan((f"enter text {self.turn}",))
            return Finish(success=True, message="done")

    config.memory.executor.blocks = config.memory.planner.blocks = ["local"]
    goals = iter(["enter text 1", "enter text 2"] if planning else ["read count"])
    child_memories = []
    contexts = []

    def factory(context):
        contexts.append(context)
        return LocalBlock(role=context.agent)

    calls = [("complete", {"success": True, "message": "done"})] * (2 if planning else 1)
    selector = Selector(*calls)
    factories = {"local": factory}
    operator = (
        PlanningOperator(LocalPlanner(), selector, memory_blocks=factories)
        if planning
        else AgentOperator(selector, memory_blocks=factories)
    )
    try:
        with ConfigLoader.override(config):
            assert (await operator.execute(Phone(), "read count")).success
    finally:
        block_registry._installed.cache_clear()
    if planning:
        first, second = child_memories
        assert first.memory_blocks[0] is not second.memory_blocks[0]
        assert [context.agent for context in contexts] == ["executor", "planner", "executor"]
        assert contexts[1].state.executor == "custom"
    else:
        assert [context.agent for context in contexts] == ["executor"]


@pytest.mark.anyio
async def test_planning_rejects_invalid_child_memory_before_device_observation(config):
    class UnusedPhone(Phone):
        async def refresh(self):
            pytest.fail("the device must not be observed with invalid memory")

    config.memory.executor.blocks = ["local"]
    # Returning a wrong name can only be detected by actually building the block.
    from mobilerun_agents.agent.memory.blocks.goal import GoalBlock

    operator = PlanningOperator(
        ScriptPlanner(), Script(), memory_blocks={"local": lambda _: GoalBlock()}
    )
    with ConfigLoader.override(config), pytest.raises(ValueError, match=r"'local'.*'goal'"):
        await operator.execute(UnusedPhone(), "read count")


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
        def bind(self, output_schema, repair):
            assert output_schema == Answer.model_json_schema()
            return self

    class TypedAgent(Script):
        async def select(self, ctx, device, memory, registry):
            schema = registry.by_name["complete"].metadata.get_parameters_dict()
            assert "output" in schema["properties"]
            return await super().select(ctx, device, memory, registry)

    config.recording.enabled = False
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
async def test_sensitive_tool_arg_is_masked_in_logs_events_and_recording(
    config, caplog, monkeypatch
):
    from mobilerun.core.devices import Device

    marker = "fixture-sensitive-text"
    logger = logging.getLogger("fixture.sensitive_tool")
    caplog.set_level(logging.INFO, logger=logger.name)
    captured = []
    acted = []
    original_event = consumers.ConsoleLog.event

    def capture_event(console, event):
        if isinstance(event, Acted):
            acted.append(event)
        return original_event(console, event)

    monkeypatch.setattr(consumers.ConsoleLog, "event", capture_event)

    @tool(sensitive_parameters=frozenset({"text"}))
    async def record_sensitive(text: str, phone: Device) -> str:
        """Receive a sensitive fixture value."""
        captured.append(text)
        logger.info("Inside declared tool: %s", text)
        await phone.input_text(text)
        return f"received {text}"

    class SensitiveScript(Script):
        async def select(self, ctx, device, memory, registry):
            selection = await super().select(ctx, device, memory, registry)
            selection.thought = f"Use {marker}"
            return selection

    selector = SensitiveScript(
        ("record_sensitive", {"text": marker}),
        ("complete", {"success": True, "message": f"received {marker}"}),
    )
    phone = Phone()
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [record_sensitive]).execute(phone, "record it")
    assert result.success and result.answer == f"received {marker}"
    assert captured == phone.inputs == [marker]
    assert [r.getMessage() for r in caplog.records if r.name == logger.name] == [
        "Inside declared tool: ***"
    ]
    action = next(event for event in acted if event.action.call.tool_name == "record_sensitive")
    assert action.action.call.tool_kwargs == {"text": "***"}
    assert marker not in action.model_dump_json() and action.step_ids == [1]
    folder = Path(result.trajectory_path)
    evidence = (folder / "agent.json").read_text()
    steps = (folder / "steps.jsonl").read_text()
    assert marker not in evidence and marker not in steps
    assert {"kind": "tool", "text": "record_sensitive", "step_ids": [1]} in json.loads(evidence)
    [row] = [json.loads(line) for line in steps.splitlines()]
    assert row["i"] == 1 and row["args"]["text"] == "***"


@pytest.mark.anyio
async def test_reused_sensitive_argument_is_not_saved_to_steps_or_macro(config):
    from mobilerun.core.devices import Device

    marker = "fixture-sensitive-reuse"
    remembered: list[str] = []

    @tool(sensitive_parameters=frozenset({"value"}))
    async def protect(value: str) -> None:
        """Receive a sensitive fixture value."""
        remembered.append(value)

    @tool
    async def type_remembered(phone: Device) -> None:
        """Type the stored fixture value without model-supplied arguments."""
        await phone.input_text(remembered[0])
        await phone.input_text("ordinary")

    selector = Script(
        ("protect", {"value": marker}),
        ("type_remembered", {}),
        ("complete", {"success": True, "message": "done"}),
    )
    phone = Phone()
    with ConfigLoader.override(config):
        result = await AgentOperator(selector, [protect, type_remembered]).execute(
            phone, "enter text"
        )
    assert result.success and phone.inputs == [marker, "ordinary"]
    folder = Path(result.trajectory_path)
    steps = (folder / "steps.jsonl").read_text()
    macro = Path(get_macro_settings().macro_directory) / folder.name / "v1.jsonl"
    assert [json.loads(line)["args"]["text"] for line in steps.splitlines()] == ["***", "ordinary"]
    assert [step.args["text"] for step in load_macro(macro)] == ["ordinary"]


@pytest.mark.anyio
async def test_custom_verification_tool_redacts_agent_evidence_and_completion_log(config, caplog):
    from mobilerun.core.devices import Device
    from mobilerun.core.extensions import action
    from mobilerun.core.mailbox import Mailbox

    class Mail(Mailbox):
        address = None

        @action
        async def wait_for_code(self) -> str:
            """Return the synthetic code."""
            return "314159"

    @tool
    async def read_mail(phone: Device) -> str:
        """Read a code directly from the mailbox extension."""
        return await phone.extensions["email"].wait_for_code()

    class Selector(Script):
        async def select(self, ctx, device, memory, registry):
            selection = await super().select(ctx, device, memory, registry)
            return Selection(calls=selection.calls, thought="Code was 314-159")

    phone = Phone()
    mail = await Mail().bind(phone)
    try:
        caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
        selector = Selector(
            ("read_mail", {}), ("complete", {"success": True, "message": "Code: 314-159"})
        )
        with ConfigLoader.override(config):
            result = await AgentOperator(selector, [read_mail]).execute(phone, "Read inbox")
    finally:
        await mail.unbind()
    assert result.answer == "Code: 314-159"
    evidence = (Path(result.trajectory_path) / "agent.json").read_text()
    assert "314-159" not in evidence and "314159" not in evidence
    assert "314-159" not in caplog.text and "314159" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["mailbox", "sms"])
@pytest.mark.parametrize("alias", [False, True])
async def test_sensitive_extension_result_stays_live_not_in_agent_evidence(
    config, caplog, kind, alias
):
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
    if alias:
        extension = type(extension)(name="verification_inbox")
        tool_name = f"{extension.name}__{tool_name.split('__', 1)[1]}"
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    model = Model(vision="off")
    await extension.bind(phone)
    try:
        with ConfigLoader.override(config):
            result = await AgentOperator(model).execute(phone, "Read a message")
    finally:
        await extension.unbind()
    assert result.success and len(model.requests) == 3 and phone.inputs == [code]
    assert result.answer == f"Code: {code}"
    assert any(secret in (message.content or "") for message in model.requests[1])
    folder = Path(result.trajectory_path)
    evidence = (folder / "agent.json").read_text()
    assert code not in evidence and not (folder / "memory").exists()
    assert code not in caplog.text
    assert code not in (folder / "steps.jsonl").read_text()
    macro_dir = Path(get_macro_settings().macro_directory) / folder.name
    assert all(code not in path.read_text() for path in macro_dir.glob("v*.jsonl"))


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_screens_remain_live_without_saving_pngs(config, planning):
    from mobilerun.core.devices import Device

    from mobilerun_agents.agent.action_selector.base import screenshot_bytes

    @tool(sensitive_parameters=frozenset({"value"}))
    async def type_private(value: str, phone: Device) -> None:
        """Accept a private value."""
        await phone.input_text(value)

    class RetainingPhone(Phone):
        async def screenshot(self, hide_overlay=True):
            return self.inputs[-1].encode() if self.inputs else b"empty screenshot"

    seen = []

    class VisionScript(Script):
        async def select(self, ctx, device, memory, registry):
            state = await ctx.store.get_state()
            seen.append(await screenshot_bytes(ctx, state.current, device))
            return await super().select(ctx, device, memory, registry)

    phone = RetainingPhone()
    finish = ("complete", {"success": True, "message": "done"})
    first = VisionScript(("type_private", {"value": "fixture-sensitive-vision"}), finish)
    with ConfigLoader.override(config):
        sensitive = await AgentOperator(first, [type_private]).execute(phone, "first run")
        selector = VisionScript(finish)
        later_operator = (
            PlanningOperator(ScriptPlanner(), selector) if planning else AgentOperator(selector)
        )
        later = await later_operator.execute(phone, "later run")
        clean = await AgentOperator(VisionScript(finish)).execute(RetainingPhone(), "fresh device")
    assert sensitive.success and later.success and clean.success
    assert seen == [
        b"empty screenshot",
        b"fixture-sensitive-vision",
        b"fixture-sensitive-vision",
        b"empty screenshot",
    ]
    for run in (sensitive, later, clean):
        assert not list(Path(run.trajectory_path).rglob("*.png"))


@pytest.mark.anyio
async def test_cancelled_batch_keeps_evidence_for_completed_call(config):
    from mobilerun.core.devices import Device

    entered = asyncio.Event()

    @tool
    async def write_first(phone: Device) -> None:
        """Write the first value."""
        await phone.input_text("first")

    @tool
    async def stall() -> None:
        """Pause before committing a second action."""
        entered.set()
        await asyncio.Future()

    class Batch(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            return Selection(
                calls=[
                    ToolSelection(tool_id=name, tool_name=name, tool_kwargs={})
                    for name in ("write_first", "stall")
                ]
            )

    config.agent.parallel_tools = True
    recorder = RecordingDevice(Phone(), capture="actions")
    with ConfigLoader.override(config):
        task = asyncio.create_task(
            AgentOperator(Batch(), [write_first, stall]).execute(recorder, "test")
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert recorder.step_index == 1
    entries = json.loads((recorder.run_dir / "agent.json").read_text())
    assert [entry for entry in entries if entry["kind"] == "tool"] == [
        {"kind": "tool", "text": "write_first", "step_ids": [1]}
    ]


@pytest.mark.anyio
async def test_overlapping_agents_link_only_their_own_tool_rows(config):
    from mobilerun.core.devices import Device, RecordingDevice

    recorder = RecordingDevice(Phone(), capture="actions")
    started = asyncio.Event()
    release = asyncio.Event()
    second_selected = asyncio.Event()

    @tool
    async def first(phone: Device) -> None:
        """Write two values with an intervening pause."""
        await phone.input_text("first-start")
        started.set()
        await release.wait()
        await phone.input_text("first-end")

    @tool
    async def second(phone: Device) -> None:
        """Write a single value."""
        await phone.input_text("second")

    class Second(Script):
        async def select(self, ctx, device, memory, registry):
            selection = await super().select(ctx, device, memory, registry)
            second_selected.set()
            return selection

    finish = ("complete", {"success": True, "message": "done"})
    with ConfigLoader.override(config):
        a = asyncio.create_task(
            AgentOperator(Script(("first", {}), finish), [first]).execute(recorder, "first")
        )
        await asyncio.wait_for(started.wait(), 5)
        b = asyncio.create_task(
            AgentOperator(Second(("second", {}), finish), [second]).execute(recorder, "second")
        )
        try:
            await asyncio.wait_for(second_selected.wait(), 5)
            right = await asyncio.wait_for(b, 5)
        finally:
            release.set()
            left = await a
    assert left.success and right.success
    entries = json.loads((recorder.run_dir / "agent.json").read_text())
    assert {
        entry["text"]: entry["step_ids"]
        for entry in entries
        if entry["kind"] == "tool" and entry["text"] != "complete"
    } == {"first": [1, 3], "second": [2]}


@pytest.mark.anyio
@pytest.mark.parametrize("child_task", [False, True])
async def test_agent_links_only_rows_committed_in_the_tool_task(config, child_task):
    from mobilerun.core.devices import Device, RecordingDevice

    recorder = RecordingDevice(Phone(), capture="actions")
    started = asyncio.Event()
    release = asyncio.Event()

    @tool
    async def write_twice(phone: Device) -> None:
        """Write around a concurrent host action."""
        await phone.input_text("first")
        started.set()
        await release.wait()
        if child_task:
            await asyncio.create_task(phone.input_text("last"))
        else:
            await phone.input_text("last")

    selector = Script(("write_twice", {}), ("complete", {"success": True, "message": "done"}))
    with ConfigLoader.override(config):
        task = asyncio.create_task(AgentOperator(selector, [write_twice]).execute(recorder, "test"))
        try:
            await asyncio.wait_for(started.wait(), 5)
            await recorder.input_text("host")
        finally:
            release.set()
            result = await task
    assert result.success
    rows = [
        json.loads(line) for line in (recorder.run_dir / "steps.jsonl").read_text().splitlines()
    ]
    assert [row["args"]["text"] for row in rows] == ["first", "host", "last"]
    entries = json.loads((recorder.run_dir / "agent.json").read_text())
    [entry] = [
        entry for entry in entries if entry["kind"] == "tool" and entry["text"] == "write_twice"
    ]
    assert entry["step_ids"] == [1, 3]


@pytest.mark.anyio
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
async def test_operator_links_memory_in_agent_json_at_any_log_level(config, caplog, level):
    from mobilerun_agents.agent.logs.memory import report_memory

    class MemorySelector(Script):
        async def select(self, ctx, device, memory, registry):
            if not device._unwrap().inputs:
                report_memory(ctx, memory, await read_memory(memory, ctx))
            return await super().select(ctx, device, memory, registry)

    caplog.set_level(level, logger="mobilerun_agents.agent.logs.console")
    selector = MemorySelector(
        ("input_text", {"text": "fixture"}), ("complete", {"success": True, "message": "done"})
    )
    with ConfigLoader.override(config):
        result = await AgentOperator(selector).execute(Phone(), "goal")
    folder = Path(result.trajectory_path)
    entries = json.loads((folder / "agent.json").read_text())
    [memory] = [entry for entry in entries if entry["kind"] == "memory"]
    assert memory["step_ids"] == [1] and "Goal: goal" in memory["text"]
    assert ("memory:" in caplog.text) is (level == logging.DEBUG)
    assert not (folder / "memory").exists()


@pytest.mark.anyio
async def test_reused_tainted_recorder_omits_first_thought(config, caplog):
    marker = "fixture-reused-taint-canary"

    class Selector(ActionSelector):
        async def select(self, ctx, device, memory, registry):
            return Selection(
                calls=[
                    ToolSelection(
                        tool_id="done",
                        tool_name="complete",
                        tool_kwargs={"success": True, "message": marker},
                    )
                ],
                thought=f"Visible text is {marker}",
            )

    recorder = RecordingDevice(Phone(), capture="actions")
    recorder._extension_state.sensitive_ui = True
    caplog.set_level(logging.DEBUG, logger="mobilerun_agents.agent.logs.console")
    with ConfigLoader.override(config):
        result = await AgentOperator(Selector()).execute(recorder, "Read the screen")
    assert result.answer == marker
    entries = json.loads((recorder.run_dir / "agent.json").read_text())
    [thought] = [entry["text"] for entry in entries if entry["kind"] == "thought"]
    assert thought == "[omitted]" and marker not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("planning", [False, True])
async def test_execution_result_keeps_raw_values_in_typed_output(config, planning):
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
    assert result.success and result.answer == message
    assert result.output == TypedOutput(details={"text": [marker, "ordinary"]})


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
    source = (Path(result.trajectory_path) / "agent.json").read_text()
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
    source = (Path(result.trajectory_path) / "agent.json").read_text()
    assert "Goal achieved." not in source


@pytest.mark.anyio
async def test_vault_action_exports_reference_without_fetching_value_into_state(config):
    class EditablePhone(Phone):
        async def refresh(self):
            return UIState([], "", "", {"focusedElement": {"text": ""}, "isEditable": True}, 0, 0)

    marker = "vault-fixture-credential"
    phone = EditablePhone()
    credentials = await DictCredentials({"app": {"sample": {"password": marker}}}).bind(phone)
    selector = Script(
        ("vault__type_field", {"package": "app", "name": "sample", "field": "password"}),
        ("complete", {"success": True, "message": "Entered saved password"}),
    )
    try:
        with ConfigLoader.override(config):
            result = await AgentOperator(selector).execute(phone, "enter the saved password")
    finally:
        await credentials.unbind()
    assert phone.inputs == [marker]
    assert result.success
    folder = Path(result.trajectory_path)
    assert marker not in (folder / "agent.json").read_text()
    macro_path = Path(get_macro_settings().macro_directory) / folder.name / "v1.jsonl"
    macro = load_macro(macro_path)
    assert marker not in macro_path.read_text()
    assert [(step.action, step.args["name"]) for step in macro] == [
        ("extension_call", "vault.type_field")
    ]


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
    config.recording.enabled = False
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


@pytest.mark.anyio
async def test_invalid_public_output_warns_without_values(config, caplog):
    class InvalidOutputWorkflow(Workflow):
        @step
        async def finish(self, ctx: Context, ev: StartEvent) -> StopEvent:
            outcome = Outcome(success=True, reason="complete", output={"count": "fixture-value"})
            ctx.write_event_to_stream(Finished(outcome=outcome, steps=1))
            return StopEvent(result=outcome)

    phone = Phone()
    result = await consumers._consume(
        config,
        InvalidOutputWorkflow(),
        None,
        phone,
        "read",
        Answer,
        1,
        ToolRegistry(phone),
    )
    assert result.success and result.output is None
    assert [
        record.getMessage() for record in caplog.records if record.name == consumers.__name__
    ] == ["Typed output omitted: output did not match the schema"]


@pytest.mark.anyio
async def test_public_output_does_not_require_contextual_serialization(config, caplog):
    marker = "fixture-contextual-output"

    class ContextualOutput(BaseModel):
        text: str = marker

        @field_serializer("text")
        def serialize_text(self, value: str, info: SerializationInfo) -> str:
            return info.context["prefix"] + value

    selector = Script(("complete", {"success": True, "message": "done", "output": {}}))
    with ConfigLoader.override(config):
        result = await AgentOperator(selector).execute(Phone(), "read", output=ContextualOutput)

    assert result.success and result.output == ContextualOutput(text=marker)
    assert not [r for r in caplog.records if r.name == consumers.__name__]


@pytest.mark.anyio
@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancellation"])
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("planning", [False, True])
async def test_run_memory_closes_after_workflow(
    config, monkeypatch, caplog, exit_kind, close_fails, planning
):
    started = asyncio.Event()
    operations = []
    used = []
    closed = []
    original_close = _RunMemory.aclose

    async def close(memory):
        operations.append("close")
        closed.append(memory)
        await original_close(memory)
        if close_fails:
            raise RuntimeError("fixture close failure")

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

    assert operations == ["workflow exited", "close"]
    assert len(used) == len(closed) == 1 and used[0] is closed[0]
    if planning:
        assert [block.name for block in used[0].memory_blocks] == config.memory.planner.blocks
    assert ("Unable to close run memory" in caplog.text) == close_fails
    assert "fixture close failure" not in caplog.text

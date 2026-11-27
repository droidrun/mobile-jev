import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import InsertMethod, Memory, StaticMemoryBlock
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import ContextItem
from sqlalchemy.ext.asyncio import AsyncEngine

from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.memory import memory as memory_module
from mobilerun_agents.agent.memory.blocks.app_cards import AppCardBlock
from mobilerun_agents.agent.memory.blocks.goal import GoalBlock
from mobilerun_agents.agent.memory.blocks.history import HistoryBlock
from mobilerun_agents.agent.memory.blocks.last_step import LastStepBlock
from mobilerun_agents.agent.memory.memory import build_executor_memory, build_planner_memory
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState, PlanningState
from mobilerun_agents.config.user_config import MemoryConfig


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("memory tests cannot open network connections")

    for method in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, method, blocked)


@pytest.fixture
def snapshot():
    return StepSnapshot(
        ui=UIState([], "", "", {"packageName": "mail", "activityName": "Inbox"}, 0, 0),
        device_context=[ContextItem(source="battery", kind="status", data={"level": 63})],
    )


def test_native_memory_never_truncates_priority_zero():
    async def check():
        memory = Memory.from_defaults(
            memory_blocks=[StaticMemoryBlock(name="pinned", static_content="keep", priority=0)],
            token_limit=2,
            tokenizer_fn=list,
        )
        assert "keep" in (await memory.aget())[0].content

    asyncio.run(check())


def test_native_memory_truncates_priority_one_before_two():
    async def check():
        memory = Memory.from_defaults(
            memory_blocks=[
                StaticMemoryBlock(name="second", static_content="keep", priority=2),
                StaticMemoryBlock(name="first", static_content="drop", priority=1),
            ],
            token_limit=4,
            tokenizer_fn=list,
        )
        text = (await memory.aget())[0].content
        assert "keep" in text and "drop" not in text

    asyncio.run(check())


@pytest.mark.parametrize("has_user", [False, True])
def test_native_memory_user_inserts_into_latest_or_new_user(has_user):
    async def check():
        memory = Memory.from_defaults(
            memory_blocks=[StaticMemoryBlock(name="setup", static_content="remember me")],
            insert_method=InsertMethod.USER,
            tokenizer_fn=list,
        )
        if has_user:
            await memory.aput_messages(
                [ChatMessage(role="user", content=text) for text in ("older", "latest")]
            )
        await memory.aput(ChatMessage(role="assistant", content="answer"))
        messages = await memory.aget()
        if has_user:
            assert [m.role for m in messages] == ["user", "user", "assistant"]
            assert messages[0].content == "older"
            assert messages[1].content.endswith("latest")
        else:
            assert [m.role for m in messages] == ["assistant", "user"]
        assert "remember me" in messages[1].content

    asyncio.run(check())


def test_native_memory_keeps_an_image_returned_with_text():
    async def check():
        image = ImageBlock(url="https://example.invalid/screen.png")
        memory = Memory.from_defaults(
            memory_blocks=[
                StaticMemoryBlock(name="mixed", static_content=[TextBlock(text="caption"), image])
            ],
            insert_method=InsertMethod.USER,
            tokenizer_fn=list,
        )
        message = (await memory.aget())[0]
        assert "caption" in message.content
        assert image in message.blocks

    asyncio.run(check())


def test_state_block_requires_state_and_redactor_and_renders_it():
    from mobilerun_agents.agent.memory.blocks import StateBlock

    class GoalBlock(StateBlock[AgentState, str]):
        async def render(self, state: AgentState, redact: Redactor) -> str:
            return state.goal

    async def check():
        block = GoalBlock(name="goal")
        assert not block.accept_short_term_memory
        with pytest.raises(TypeError, match="requires.*state"):
            await block.aget()
        with pytest.raises(TypeError, match="requires.*redact"):
            await block.aget(state=AgentState(goal="Find order"))
        assert (
            await block.aget(state=AgentState(goal="Find order"), redact=Redactor()) == "Find order"
        )

    asyncio.run(check())


def test_read_memory_renders_the_latest_persisted_state():
    from mobilerun_agents.agent.memory import read_memory

    async def check():
        first = AgentState(goal="Find order")
        second = AgentState(goal="Find receipt")
        ctx = SimpleNamespace(
            store=SimpleNamespace(get_state=AsyncMock(side_effect=[first, second]))
        )
        memory = build_executor_memory(MemoryConfig(), "android")
        try:
            redact = Redactor()
            assert "Find order" in (await read_memory(memory, ctx, redact))[0].content
            messages = await read_memory(memory, ctx, redact)
            assert "Find receipt" in messages[0].content and "Find order" not in messages[0].content
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_setup_helper_builds_pinned_static_memory():
    from mobilerun_agents.agent.memory.memory import build_setup_block

    block = build_setup_block("setup instructions")
    assert (block.name, block.priority) == ("setup", 0)
    assert asyncio.run(block.aget()) == [TextBlock(text="setup instructions")]


def test_goal_renders_planning_state_and_last_step_is_empty_before_acting():
    assert asyncio.run(
        GoalBlock().render(PlanningState(goal="Find order", notes=["Pending"]), Redactor())
    ) == ("Goal: Find order\nPending")
    assert asyncio.run(LastStepBlock().render(AgentState(), Redactor())) == []


def test_history_window_groups_calls_and_truncates_oldest_first(snapshot):
    async def check():
        snapshot.ui.elements.append({"index": 19, "text": "Search on eBay", "className": "Button"})
        actions = [
            ActionRecord(
                step=step,
                call=ToolSelection(tool_id=str(step), tool_name=name, tool_kwargs={"index": 19}),
                ok=ok,
                summary="not the compact label",
                before=snapshot,
                screen_changed=True,
            )
            for step, name, ok in [
                (1, "open", True),
                (2, "swipe", True),
                (3, "tap_element", True),
                (3, "wait", False),
                (4, "complete", True),
            ]
        ]
        block = HistoryBlock(steps=2)
        text = await block.render(AgentState(actions=actions), Redactor())
        older, newer = text.splitlines()
        assert older.startswith("2 · swipe")
        assert (
            newer
            == '3 · tap_element [19] Button "Search on eBay" ✓: not the compact label · wait [19] Button "Search on eBay" ✗: not the compact label · screen changed'
        )
        assert await block.atruncate(text, 1) == newer
        assert await block.atruncate(text, 1000) == ""

    asyncio.run(check())


@pytest.mark.parametrize("screenshot_enabled", [True, False])
def test_executor_config_and_last_step_content(snapshot, screenshot_enabled, tmp_path):
    async def check():
        config = MemoryConfig(
            executor={
                "blocks": ["setup", "last_step", "goal"],
                "last_step_screenshot": screenshot_enabled,
            }
        )
        memory = build_executor_memory(config, "android")
        snapshot.ui.phone_state.update(focusedElement={"text": "Search"}, keyboardVisible=True)
        path = tmp_path / "before.png"
        path.write_bytes(b"fake screenshot")
        snapshot.screenshot = ImageBlock(path=path)
        after = StepSnapshot(
            ui=UIState(
                [],
                "",
                "",
                {
                    "packageName": "browser",
                    "activityName": "Results",
                    "keyboardVisible": False,
                },
                0,
                0,
            )
        )
        actions = [
            ActionRecord(
                step=step,
                call=ToolSelection(tool_id=name, tool_name=name, tool_kwargs={"index": 19}),
                ok=ok,
                summary=summary,
                thought="Find the order",
                before=snapshot,
                screen_changed=True,
            )
            for step, name, ok, summary in [
                (1, "open", True, "Old result"),
                (2, "tap_element", True, "Opened results"),
                (2, "wait", False, "Timed out"),
            ]
        ]
        ctx = SimpleNamespace(
            store=SimpleNamespace(
                get_state=AsyncMock(
                    return_value=AgentState(
                        goal="Find order",
                        notes=["Order is pending"],
                        current=after,
                        actions=actions,
                    )
                )
            )
        )
        try:
            assert [block.name for block in memory.memory_blocks] == ["setup", "last_step", "goal"]
            (message,) = await memory_module.read_memory(memory, ctx, Redactor())
            text = message.content
            assert [text.index(f"<{name}>") for name in ("setup", "last_step", "goal")] == sorted(
                text.index(f"<{name}>") for name in ("setup", "last_step", "goal")
            )
            for part in (
                'tap_element({"index": 19})',
                "ok=true",
                "ok=false",
                "Opened results",
                "Timed out",
                "Find the order",
                "mail",
                "Inbox",
                "browser",
                "Results",
                "Search",
                "true → false",
                "Screen changed: true",
                "Order is pending",
            ):
                assert part in text
            assert "Old result" not in text and "<history>" not in text
            assert (snapshot.screenshot in message.blocks) is screenshot_enabled
            print(text)
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_shared_engine_keeps_sessions_separate_and_is_caller_owned():
    async def check():
        engine = memory_module._private_engine()
        try:
            config = MemoryConfig()
            first = build_executor_memory(
                config, "android", async_engine=engine, session_id="first"
            )
            second = build_executor_memory(
                config, "android", async_engine=engine, session_id="second"
            )
            assert (first.session_id, second.session_id) == ("first", "second")
            await first.aput(ChatMessage(role="user", content="first only"))
            await second.aput(ChatMessage(role="user", content="second only"))
            assert [m.content for m in await first.aget_all()] == ["first only"]
            await first.aclose()
            assert [m.content for m in await second.aget_all()] == ["second only"]
        finally:
            await engine.dispose()

    asyncio.run(check())


def test_repeat_uses_the_matching_snapshot_when_two_actions_share_a_step(snapshot):
    snapshot.ui.elements.append({"index": 3, "text": "first"})
    other = snapshot.model_copy(update={"ui": UIState([{"index": 4}], "", "", {}, 0, 0)})
    calls = [
        ToolSelection(tool_id=str(i), tool_name=name, tool_kwargs={"index": i})
        for i, name in [(3, "tap"), (4, "long_press")]
    ]
    actions = [
        ActionRecord(step=1, call=call, ok=True, summary="ok", before=before)
        for call, before in zip(calls, (snapshot, other), strict=True)
    ]
    state = AgentState(current=snapshot, actions=actions)

    async def check():
        memory = build_executor_memory(MemoryConfig(), "android")
        for current, expected, unexpected in [
            (snapshot, "tap", "long_press"),
            (other, "long_press", "tap"),
        ]:
            state.current = current
            messages = await memory.aget(state=state, redact=Redactor())
            hints = "\n".join(m.content or "" for m in messages)
            assert f"Last action here: {expected}" in hints
            assert f"Last action here: {unexpected}" not in hints

    asyncio.run(check())


@pytest.mark.parametrize(
    ("platform", "executor", "expected"),
    [
        ("android", None, True),
        ("ios", None, True),
        ("web", None, False),
        ("android", {"blocks": ["setup"]}, False),
    ],
)
def test_app_cards_follow_executor_block_list_on_supported_platforms(platform, executor, expected):
    config = MemoryConfig.model_validate({"executor": executor} if executor else {})
    memory = build_executor_memory(config, platform)
    assert any(block.name == "app_cards" for block in memory.memory_blocks) is expected
    asyncio.run(memory.aclose())


@pytest.mark.parametrize("method", list(InsertMethod))
def test_blocks_render_notes_but_not_current_device_context(snapshot, method):
    state = AgentState(goal="Find order", notes=["Order is pending"], current=snapshot)
    memory = build_executor_memory(MemoryConfig(insert_method=method), "android")

    async def check():
        redact = Redactor()
        result = await memory.aget(state=state, redact=redact)
        text = result[0].content
        assert result[0].role.value == method.value
        assert "Find order" in text and "battery" not in text
        assert text.count("Order is pending") == 1
        state.notes.clear()
        assert "Order is pending" not in (await memory.aget(state=state, redact=redact))[0].content
        assert await memory.aget_all() == []

    asyncio.run(check())


def test_default_memory_budget():
    memory = build_executor_memory(MemoryConfig(), "android")
    assert (memory.token_limit, memory.chat_history_token_ratio) == (30000, 0)


def test_executor_first_turn_renders_pinned_memory_under_token_pressure():
    async def check():
        memory = build_executor_memory(MemoryConfig(token_limit=1), "android")
        try:
            (message,) = await memory.aget(state=AgentState(goal="Find order"), redact=Redactor())
            assert "Goal: Find order" in message.content
            assert "<history>" not in message.content
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_historical_ui_labels_are_masked_in_executor_memory(snapshot):
    async def check():
        secret = "private-recipient"
        redact = Redactor()
        redact.add(secret)
        snapshot.ui.elements.append({"index": 1, "text": secret})
        snapshot.ui.phone_state["focusedElement"] = {"text": secret}
        current = StepSnapshot(ui=UIState([], "Inbox", "", {}, 0, 0))
        actions = [
            ActionRecord(
                step=step,
                call=ToolSelection(
                    tool_id=str(step), tool_name="tap_element", tool_kwargs={"index": 1}
                ),
                ok=True,
                summary="Tapped",
                before=snapshot,
            )
            for step in (1, 2)
        ]
        state = AgentState(current=current, actions=actions)
        memory = build_executor_memory(MemoryConfig(), "android")
        try:
            (message,) = await memory.aget(state=state, redact=redact)
            assert secret not in message.content
            assert 'Focused element: Element "***"' in message.content
            assert message.content.count('Element "***"') >= 2
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_history_keeps_masked_arguments_and_compact_lookup_result(snapshot):
    async def check():
        redact = Redactor()
        redact.add("private-recipient")
        prefix = "Tracking number: 123. " + "a" * 92
        calls = [
            (
                "lookup",
                {"query": "order", "account": "private-recipient"},
                prefix + "private-recipient" + "unneeded tail",
            ),
            ("wait", {}, "Waited"),
        ]
        state = AgentState(
            actions=[
                ActionRecord(
                    step=step,
                    call=ToolSelection(tool_id=str(step), tool_name=name, tool_kwargs=args),
                    ok=True,
                    summary=summary,
                    before=snapshot,
                )
                for step, (name, args, summary) in enumerate(calls, 1)
            ]
        )
        memory = build_executor_memory(MemoryConfig(), "android")
        try:
            (message,) = await memory.aget(state=state, redact=redact)
            assert '"query":"order"' in message.content and '"account":"***"' in message.content
            assert prefix.rstrip() in message.content
            assert "private" not in message.content and "unneeded tail" not in message.content
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_executor_truncates_app_cards_before_action_history(snapshot):
    async def check():
        snapshot.ui.phone_state["packageName"] = "com.google.android.gm"
        state = AgentState(
            current=snapshot,
            actions=[
                ActionRecord(
                    step=step,
                    call=ToolSelection(tool_id=str(step), tool_name="wait", tool_kwargs={}),
                    ok=True,
                    summary="Waited",
                    before=snapshot,
                )
                for step in (1, 2)
            ],
        )
        memory = build_executor_memory(MemoryConfig(token_limit=300), "android")
        try:
            (message,) = await memory.aget(state=state, redact=Redactor())
            assert "<app_cards>" not in message.content
            assert "1 · wait" in message.content
        finally:
            await memory.aclose()

    asyncio.run(check())


def test_local_card_mapping_cannot_escape_platform_folder(snapshot, tmp_path):
    folder = tmp_path / "android"
    folder.mkdir()
    (tmp_path / "outside.md").write_text("private")
    (folder / "app_cards.json").write_text('{"mail":"../outside.md"}')
    block = AppCardBlock(platform="android", cards_dir=tmp_path)
    assert asyncio.run(block.aget(state=AgentState(current=snapshot), redact=Redactor())) == ""


def test_local_card_uses_platform_and_caches_by_package(snapshot, tmp_path):
    for platform in ("ios", "android"):
        folder = tmp_path / platform
        folder.mkdir()
        (folder / "app_cards.json").write_text('{"mail":"mail.md"}')
        (folder / "mail.md").write_text(f"{platform} guide")

    block = AppCardBlock(platform="ios", cards_dir=tmp_path)

    async def check():
        state = AgentState(goal="Find order", current=snapshot)
        redact = Redactor()
        assert await block.aget(state=state, redact=redact) == "ios guide"
        (tmp_path / "ios" / "mail.md").write_text("updated guide")
        state.goal = "Find receipt"
        assert await block.aget(state=state, redact=redact) == "ios guide"
        snapshot.ui.phone_state["packageName"] = "browser"
        assert await block.aget(state=state, redact=redact) == ""
        snapshot.ui.phone_state.pop("packageName")
        assert await block.aget(state=state, redact=redact) == ""

    asyncio.run(check())


@pytest.mark.parametrize("planner", [False, True])
def test_builders_share_the_prompt_template_and_private_engine_cleanup(monkeypatch, planner):
    engine = Mock(spec=AsyncEngine)
    monkeypatch.setattr(memory_module, "_private_engine", lambda: engine)
    config = MemoryConfig()
    memory = (
        build_planner_memory(config, PlanningState())
        if planner
        else build_executor_memory(config, "android")
    )

    asyncio.run(memory.aclose())

    engine.dispose.assert_awaited_once_with()
    assert type(memory.memory_blocks_template) is memory_module._BlockPrompt

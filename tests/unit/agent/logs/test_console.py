import asyncio
import logging

import click
import pytest

from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.diagnostics import (
    ActionDetail,
    Detail,
    ModelCall,
    Screen,
    private_values,
)
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.state import ActionOutcome
from mobilerun_agents.errors import SelectorCannotServe, UnknownModel


def test_console_log_formats_event_summary_without_sensitive_fields(caplog):
    consumer = ConsoleLog()
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")

    consumer.on_repair()
    consumer.on_action(
        ActionOutcome(
            step=1,
            name="type",
            args={"text": "password"},
            ok=True,
            summary="Text entered",
            screenshot="full-image-data",
            ui_state=None,
        )
    )
    consumer.on_result(
        AgentResult[None](success=True, reason="Finished", answer="Done", output=None, steps=1)
    )

    output = caplog.text
    assert "Retrying invalid selection" in output
    assert "1 · type ✓" in output
    assert "✓ Done (1 step)" in output
    assert "password" not in output
    assert "full-image-data" not in output
    assert "sk-secret" not in output


def test_console_progress_order_and_safe_details(caplog):
    console = ConsoleLog()
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")

    def record(step, name, args, summary):
        return ActionOutcome(
            step=step,
            name=name,
            args=args,
            summary=summary,
            ok=True,
            screenshot="image-data",
            ui_state="private-tree",
        )

    console.on_action(record(1, "start_app", {"package": "Settings", "token": "private"}, ""))
    console.flush_step()
    console.on_subgoal("Read\nBattery")
    console.on_action(
        record(
            2,
            "tap_element",
            {"index": 12},
            "tap_element {'index': 12} → None",
        )
    )
    console.on_action(record(3, "start_app", {"package": "private"}, "[REDACTED]"))
    console.flush_step()
    console.on_repair()
    console.on_result(
        AgentResult[None](success=False, reason="Stopped", answer="unused", steps=3, output=None),
        "runs/123",
    )
    assert caplog.messages == [
        "1 · start_app Settings ✓",
        "→ subgoal: Read Battery",
        "2 · tap_element [12] ✓",
        "3 · start_app *** ✓",
        "Retrying invalid selection",
        "✗ Stopped (3 steps · runs/123)",
    ]
    assert caplog.records[-2].color == "dim"


def test_console_exception_omits_raw_request(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    ConsoleLog().on_error(RuntimeError("private request"), 0)
    assert caplog.messages == ["✗ RuntimeError (0 steps)"]


def test_console_formats_cancelled_run(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    ConsoleLog().on_error(asyncio.CancelledError(), 4, "runs/123")
    assert caplog.messages == ["✗ cancelled (4 steps · runs/123)"]


@pytest.mark.parametrize("debug", [False, True])
def test_console_marks_every_action_in_a_delegated_step(caplog, debug):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog()

    def action(name, args):
        return ActionOutcome(
            step=13,
            name=name,
            args=args,
            ok=True,
            summary="ok",
            screenshot=None,
            ui_state=None,
        )

    token = private_values.set(set() if debug else None)
    try:
        console.on_action(action("tap_element", {"index": 16, "label": "Search"}))
        console.on_action(action("input_text", {"text": "iPhone 15 Pro Max"}), delegated=True)
        console.on_action(action("tap", {"x": 561, "y": 252}), delegated=True)
        console.on_action(action("wait", {}).model_copy(update={"step": 14}))
        console.flush_step()
    finally:
        private_values.reset(token)

    assert caplog.messages == [
        "13 · tap_element [16] Search ✓ · llm",
        '13 · input_text "iPhone 15 Pro Max" ✓ · llm',
        "13 · tap (561, 252) ✓ · llm",
        "14 · wait ✓",
    ]


def test_console_masks_sensitive_typed_text(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog()
    console.on_action(
        ActionOutcome(
            step=1,
            name="input_text",
            args={"text": "***"},
            ok=True,
            summary="ok",
            screenshot=None,
            ui_state=None,
        ),
        delegated=True,
    )
    console.flush_step()
    assert caplog.messages == ['1 · input_text "***" ✓ · llm']


@pytest.mark.parametrize(
    "error",
    [
        SelectorCannotServe("Jev can't plan: use an LLM selector as the planner."),
        UnknownModel("openai", "gpt-6-sol"),
    ],
)
def test_console_shows_own_error_messages(caplog, error):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    ConsoleLog().on_error(error, 0)
    assert caplog.messages == [f"✗ {error} (0 steps)"]


def test_console_debug_renders_missing_image(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        ConsoleLog().on_event(Detail(section="image", text="no image"))
        assert capsys.readouterr().err.strip() == "image: no image"


def test_first_screen_is_visible_without_an_attempt_header(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog()
        console.on_event(Screen(package="app", activity="Main", elements=2))
        assert capsys.readouterr().err.splitlines() == [
            "  screen: app/Main · 2 elements",
        ]
        console.on_event(
            Detail(
                section="thought",
                text="  Open\n  <add_memory>one</add_memory>   the\tapp. <add_memory>two</add_memory>",
            )
        )
        assert capsys.readouterr().err.strip() == "thought: Open the app."
        console.on_error(RuntimeError("private"), 0, "runs/123")
        assert "run folder=runs/123" in capsys.readouterr().err


def test_debug_log_format_regressions(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog()
        console.on_event(Screen(package="com.android.settings", activity=None, elements=0))
        console.on_event(Detail(section="thought", text="Open Settings. Open Settings."))
        console.on_event(Detail(section="planning", text="executor: llm:openai/gpt-5"))
        console.on_event(Detail(section="planning", text="executor: llm:openai/gpt-5"))
        console.on_event(
            ModelCall(
                name="llm:openai/gpt-5",
                image=None,
                seconds=3.445,
                inputs=1,
                outputs=1,
                reasoning=0,
                total=2,
            )
        )
        console.on_event(
            ActionDetail(
                wire="tap_element",
                registry="tap_element",
                args={"index": 21},
                success=True,
                summary="tap_element {'index': 21} → None",
                seconds=0.458,
            )
        )
        console.on_action(
            ActionOutcome(
                step=1,
                name="tap_element",
                args={"index": 21},
                ok=True,
                summary="Tapped",
                screenshot=None,
                ui_state=None,
            )
        )
        console.flush_step()
        lines = capsys.readouterr().err.splitlines()
        assert lines.count("  thought: Open Settings.") == 1
        assert "  screen: com.android.settings · 0 elements" in lines
        assert sum(line == "  planning: executor: llm:openai/gpt-5" for line in lines) == 1
        assert "  LLM: llm:openai/gpt-5 · 3.445s · input=1 output=1 reasoning=0" in lines
        assert "  result: success · 0.458s" in lines


def test_long_thought_deduplicates_before_truncation(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        text = "Amazon returned a unique query product price. " + " ".join(
            f"detail{index} value{index}" for index in range(50)
        )
        repeated_prefix = text[: text.rfind(" ", 0, len(text) // 2)]
        ConsoleLog().on_event(Detail(section="thought", text=text + " " + repeated_prefix))
        line = capsys.readouterr().err.strip()
        assert " ".join(line.split()) == "thought: " + text.strip()[:300] + "…"


@pytest.mark.parametrize(
    ("reasoning", "message", "expected"),
    [
        (
            "Amazon's search field is active. Find the price from the results.",
            "Amazon's search field is active. Find the price from the results.",
            "Amazon's search field is active. Find the price from the results.",
        ),
        (
            "I'll ensure the search field remains focused before entering the query. The eBay search field is already focused. I'll search",
            "The eBay search field is already focused. I'll search for the item.",
            "I'll ensure the search field remains focused before entering the query. The eBay search field is already focused. I'll search for the item.",
        ),
    ],
)
def test_debug_thought_deduplicates_reasoning_and_message(capsys, reasoning, message, expected):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        ConsoleLog().on_event(Detail(section="thought", text=f"{reasoning} {message}"))
        assert " ".join(capsys.readouterr().err.split()) == "thought: " + expected


@pytest.mark.parametrize("middle", ["\n\t", "  "])
def test_thought_deduplicates_whitespace_and_containment(capsys, middle):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        phrase = "Amazon's search field is active. Find the price from the results."
        full = "First check the app. " + phrase + " Then compare offers."
        ConsoleLog().on_event(
            Detail(section="thought", text=full + middle + phrase.replace(" ", "  "))
        )
        assert " ".join(capsys.readouterr().err.split()) == "thought: " + full


def test_unknown_screen_package_uses_friendly_label(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        ConsoleLog().on_event(Screen(package=None, activity=None, elements=27))
        assert capsys.readouterr().err.splitlines() == [
            "  screen: unknown app · 27 elements",
        ]


def test_failed_completion_redacts_and_caps_answer(caplog):
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")
    ConsoleLog().on_result(
        AgentResult[None](
            success=False,
            reason="complete",
            answer="token=private " + "x" * 300,
            output=None,
            steps=2,
        )
    )
    assert caplog.messages == ["✗ " + ("token=*** " + "x" * 300)[:200] + "… (2 steps)"]

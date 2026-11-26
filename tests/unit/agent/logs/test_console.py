import logging

import click
import pytest

from mobilerun_agents.agent.logs import configure_cli_logging
from mobilerun_agents.agent.logs.console import ConsoleLog
from mobilerun_agents.agent.logs.diagnostics import ActionDetail, Detail, ModelCall, Screen
from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.agent.workflows.core.state import ActionRecord
from mobilerun_agents.errors import SelectorCannotServe, UnknownModel


def test_console_log_formats_event_summary_without_sensitive_fields(caplog):
    consumer = ConsoleLog()
    caplog.set_level(logging.INFO, logger="mobilerun_agents.agent.logs.console")

    consumer.on_repair()
    consumer.on_action(
        ActionRecord(
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
        return ActionRecord(
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


def test_attempt_is_visible_before_the_model_returns(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog()
        console.on_event(Screen(package="app", activity="Main", elements=2))
        assert capsys.readouterr().err.splitlines() == [
            "1 · attempt 1",
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
            ActionRecord(
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

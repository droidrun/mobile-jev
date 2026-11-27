import click
import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.events import Acted, Detail, Finished, Observed, Repaired, Subgoal
from mobilerun_agents.agent.logs import ConsoleLog, configure_cli_logging
from mobilerun_agents.agent.logs.memory import MemoryDetail
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, Outcome


def redact(text):
    return text.replace("fixture secret", "***").replace("fixture+secret", "***")


def test_console_logs_event_values_with_single_line_messages(caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(redact)
    console.event(Subgoal(text="enter\nfixture secret"))
    console.event(Repaired(step=1, attempt=1, reason="fixture secret"))
    console.event(
        Acted(
            action=ActionRecord(
                step=1,
                ok=True,
                summary="typed fixture secret",
                call=ToolSelection(
                    tool_id="1", tool_name="input_text", tool_kwargs={"text": "fixture+secret"}
                ),
                before=StepSnapshot(ui=UIState([], "", "", {}, 0, 0)),
            )
        )
    )
    console.finish(True, "done with fixture secret", 1)
    assert "→ subgoal: enter ***" in caplog.messages
    assert "Retrying invalid selection" in caplog.messages
    assert "1 · input_text ✓" in caplog.messages
    assert "  arguments: {'text': '***'} · result: typed ***" in caplog.messages
    assert "✓ done with *** (1 step)" in caplog.messages


def test_cli_routes_debug_to_stderr_and_result_to_stdout(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog(redact)
        console.event(Detail(section="image", text="no image"))
        console.finish(True, "done", 1)
    output = capsys.readouterr()
    assert "image: no image" in output.err
    assert "✓ done (1 step)" in output.out


def test_agent_console_keeps_step_count_and_path(caplog, tmp_path):
    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    finished = Finished(outcome=Outcome(success=True, reason="completed"), steps=1)
    ConsoleLog(redact).finish(True, "done", finished, tmp_path)

    assert caplog.messages == [f"✓ done (1 step · {tmp_path})"]


def test_console_distinguishes_planner_and_executor_screens(caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(redact)
    console.event(Observed(step=0, package="shop", planner=True))
    console.event(Observed(step=1, package="shop"))

    assert caplog.messages == ["  planner screen: shop", "  screen: shop"]


def test_console_subgoal_shows_shared_step_progress(caplog):
    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    ConsoleLog(redact).event(Subgoal(text="Open settings", step=12, budget=60))

    assert caplog.messages == ["→ subgoal (step 12/60): Open settings"]


def test_console_finish_shows_shared_step_breakdown(caplog, tmp_path):
    caplog.set_level("INFO", logger="mobilerun_agents.agent.logs.console")
    finished = Finished(
        outcome=Outcome(success=True, reason="completed"), steps=16, planner_turns=10, budget=60
    )
    ConsoleLog(redact).finish(True, "done", finished, tmp_path)

    assert caplog.messages == [f"✓ done (16/60 steps · 10 planner, 6 actions · {tmp_path})"]


def test_debug_memory_is_console_only(capsys, tmp_path, monkeypatch):
    console_dir = tmp_path / "console"
    console_dir.mkdir()
    monkeypatch.chdir(console_dir)
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog(redact)
        console.event(Observed(step=3))
        for text in ("x" * 140, "second memory read"):
            console.event(
                MemoryDetail(
                    section="memory",
                    text=text,
                    blocks={"notes": text},
                    last_steps=[],
                    counts="1 msg",
                )
            )
        console.finish(True, "done", 3)
    lines = capsys.readouterr().err.splitlines()
    previews = [line for line in lines if "notes:" in line]
    assert previews == [f"    notes: {'x' * 119}…"]
    assert any("memory (jev): 1 msg" in line for line in lines)
    assert not list(console_dir.iterdir())


def test_console_compacts_later_memory_details(caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(redact)
    console.event(Observed(step=2))
    console.event(
        MemoryDetail(
            section="memory",
            text="first",
            blocks={"notes": "first"},
            last_steps=["first step"],
            counts="6 msgs · 113/30k tokens",
            selector="jev",
        )
    )
    console.event(
        MemoryDetail(
            section="memory",
            text="different",
            blocks={"notes": "different"},
            last_steps=["other step"],
            counts="6 msgs · 166/30k tokens",
            selector="delegate",
        )
    )
    assert sum("memory: 6 msgs" in message for message in caplog.messages) == 1
    assert "memory (delegate): 6 msgs · 166/30k tokens" in caplog.text
    assert "different" not in caplog.text


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_console_sanitizes_debug_memory(caplog, level):
    caplog.set_level(level, logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(redact)
    console.event(
        MemoryDetail(
            section="memory",
            text="fixture secret",
            blocks={"notes": "fixture secret"},
            last_steps=[],
            counts="1 msg",
        )
    )
    console.finish(True, "done", 1)

    assert "✓ done (1 step)" in caplog.messages
    assert "fixture secret" not in caplog.text
    assert ("notes: ***" in caplog.text) is (level == "DEBUG")

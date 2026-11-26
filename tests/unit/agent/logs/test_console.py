import os

import click
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.events import Acted, Detail, Observed, Repaired, Subgoal
from mobilerun_agents.agent.logs import ConsoleLog, configure_cli_logging
from mobilerun_agents.agent.logs.memory import MemoryDetail
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord


def test_console_redacts_event_text_and_arguments(caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    redact = Redactor()
    redact.add("fixture secret")
    console = ConsoleLog(redact)
    console.event(Subgoal(text="enter fixture secret"))
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
    assert "✓ done with *** (1 step)" in caplog.messages
    assert "fixture" not in caplog.text


def test_cli_routes_debug_to_stderr_and_result_to_stdout(capsys):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        console = ConsoleLog(Redactor())
        console.event(Detail(section="image", text="no image"))
        console.finish(True, "done", 1)
    output = capsys.readouterr()
    assert "image: no image" in output.err
    assert "✓ done (1 step)" in output.out


def test_memories_share_a_step_file_without_wrapping_console_output(capsys, tmp_path):
    with click.Context(click.Command("agent")):
        configure_cli_logging(True)
        redact = Redactor()
        long_secret = "clipped-secret" * 12
        redact.add(long_secret)
        console = ConsoleLog(redact, tmp_path)
        console.event(Observed(step=3))
        for text in ("x" * 140, long_secret):
            console.event(
                MemoryDetail(
                    section="memory",
                    text=text,
                    blocks={"notes": text},
                    last_steps=[],
                    counts="1 msg",
                )
            )
        redact.add("x" * 140)
        console.finish(True, "done", 3)
    lines = capsys.readouterr().err.splitlines()
    assert sum("Memory details:" in line for line in lines) == 1
    previews = [line for line in lines if "notes:" in line]
    assert previews == [f"    notes: {'x' * 119}…"]
    assert any("memory (jev): 1 msg" in line for line in lines)
    assert "clipped-secret" not in "\n".join(lines)
    assert (tmp_path / "memory/0003.txt").read_text() == "***\n"
    assert (tmp_path / "memory/0003-jev.txt").read_text() == "***\n"


def test_console_compacts_later_memory_details_and_keeps_separate_files(caplog, tmp_path):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(Redactor(), tmp_path)
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
    assert (tmp_path / "memory/0002.txt").read_text() == "first\n"
    assert (tmp_path / "memory/0002-delegate.txt").read_text() == "different\n"


def test_memory_files_use_private_mode_independent_of_umask(tmp_path, caplog):
    caplog.set_level("DEBUG", logger="mobilerun_agents.agent.logs.console")
    console = ConsoleLog(Redactor(), tmp_path)
    previous = os.umask(0)
    try:
        console.event(
            MemoryDetail(section="memory", text="full", blocks={}, last_steps=[], counts="counts")
        )
    finally:
        os.umask(previous)
    assert (tmp_path / "memory/0001.txt").stat().st_mode & 0o777 == 0o600

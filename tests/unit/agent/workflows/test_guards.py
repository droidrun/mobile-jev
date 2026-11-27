import re
from itertools import pairwise
from typing import Literal

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.memory.blocks.history import HistoryBlock
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.agent.workflows.guards import Stop, check
from mobilerun_agents.config.user_config import AgentConfig


@pytest.mark.parametrize(
    ("screens", "names", "max_steps", "reason"),
    [
        ("AB", "a", 1, "max_steps"),
        ("AAAAA", "aaaa", 25, None),
        ("AAAAAA", "abcde", 25, "stuck"),
        ("AAAAA", "abcd", 25, None),
        ("ABABABA", "abcdef", 25, None),
        ("ABABABABA", "abcdefgh", 25, "cycle"),
        ("ABABABACA", "abcdefgh", 25, None),
    ],
)
def test_guards(screens, names, max_steps, reason):
    snapshots = [
        StepSnapshot(ui=UIState([{"text": screen}], "", "", {}, 10, 10), device_context=[])
        for screen in screens
    ]
    state = AgentState(goal="Find the screen", max_steps=max_steps, current=snapshots[-1])
    for step, (before, after), name in zip(
        range(1, len(screens)), pairwise(snapshots), names, strict=True
    ):
        state.actions.append(
            ActionRecord(
                step=step,
                call=ToolSelection(tool_id=str(step), tool_name=name, tool_kwargs={}),
                ok=True,
                summary="done",
                thought=None,
                before=before,
                screen_changed=before.fingerprint != after.fingerprint,
            )
        )
    stopped = check(state, AgentConfig())
    assert (stopped.reason if stopped else None) == reason


@pytest.mark.anyio
@pytest.mark.parametrize("variation", ["same", "index", "target", "argument", "tool"])
async def test_zip_sheet_repeat_ignores_carousel_changes(variation):
    state = AgentState(max_steps=25)
    config = AgentConfig()
    warning = 'tap_element on TextView "Enter a US zip code" repeated 3×'

    def snapshot(n):
        return StepSnapshot(
            ui=UIState(
                [
                    {"index": 0, "text": f"Carousel frame {n}"},
                    {
                        "index": 26 + n if variation == "index" else 26,
                        "className": "android.widget.TextView",
                        "text": (
                            "Search" if variation == "target" and n % 2 else "Enter a US zip code"
                        ),
                    },
                ],
                "",
                "",
                {},
                100,
                100,
            )
        )

    for step in range(1, 6):
        before, after = snapshot(step), snapshot(step + 1)
        args = {"index": 26 + step if variation == "index" else 26}
        if variation == "argument":
            args["duration_ms"] = 200 + step % 2
        state.actions.append(
            ActionRecord(
                step=step,
                call=ToolSelection(
                    tool_id=str(step),
                    tool_name=(
                        "long_press_element" if variation == "tool" and step % 2 else "tap_element"
                    ),
                    tool_kwargs=args,
                ),
                ok=True,
                summary="done",
                before=before,
                screen_changed=before.compare(after),
            )
        )
        state.current = after
        assert state.actions[-1].screen_changed is True
        hints = await HistoryBlock(config=config).render(state)
        if variation in {"same", "index"} and step >= 3:
            assert (
                warning.replace("3×", f"{step}×")
                + " — it isn't working; try a different element or action"
                in hints
            )
        else:
            assert "repeated" not in hints
        assert check(state, config) == (
            Stop("stuck") if step == 5 and variation in {"same", "index"} else None
        )


def test_target_uses_before_snapshot_not_element_index():
    from mobilerun_agents.agent.workflows.core.targets import target

    for index in (26, 41):
        record = ActionRecord(
            step=1,
            call=ToolSelection(
                tool_id="tap", tool_name="tap_element", tool_kwargs={"index": index}
            ),
            ok=True,
            summary="done",
            before=StepSnapshot(
                ui=UIState(
                    [
                        {
                            "index": index,
                            "className": "android.widget.TextView",
                            "text": "Enter a US zip code",
                        }
                    ],
                    "",
                    "",
                    {},
                    100,
                    100,
                )
            ),
        )
        assert target(record) == 'TextView "Enter a US zip code"'


def _history(screens: str, calls_per_step: tuple[int, ...]) -> AgentState:
    snapshots = [
        StepSnapshot(ui=UIState([{"text": screen}] if screen != "?" else [], "", "", {}, 10, 10))
        for screen in screens
    ]
    return AgentState(
        current=snapshots[-1],
        max_steps=25,
        actions=[
            ActionRecord(
                step=step,
                call=ToolSelection(
                    tool_id=f"{step}-{call}", tool_name="tap", tool_kwargs={"x": step}
                ),
                ok=True,
                summary="done",
                before=before,
                screen_changed=before.compare(after),
            )
            for step, ((before, after), calls) in enumerate(
                zip(pairwise(snapshots), calls_per_step, strict=True), start=1
            )
            for call in range(calls)
        ],
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("screens", "calls_per_step", "reason", "expected"),
    [
        ("AA", (3,), "stuck", 1),
        ("AAAA", (3, 1, 2), "stuck", 3),
        ("ABAA", (1, 1, 2), "stuck", 1),
        ("AA?AA", (2, 1, 1, 2), "stuck", 1),
        ("AA?", (1, 2), "stuck", 0),
        ("ABABA", (3, 2, 1, 3), "cycle", 2),
        ("ABABA?BA", (2, 1, 1, 2, 1, 2, 1), "cycle", 1),
        ("ABA?ABABA", (1,) * 8, "cycle", 3),
        ("ABABACA", (1,) * 6, "cycle", 1),
        ("ABABA?", (1,) * 5, "cycle", 0),
    ],
)
async def test_progress_note_and_guard_count_the_same_history(
    screens: str, calls_per_step: tuple[int, ...], reason: Literal["stuck", "cycle"], expected: int
):
    state = _history(screens, calls_per_step)
    common = {"stuck_fail_after": 25}
    if reason == "stuck":
        common.update(no_progress_note_after=1, cycle_note_after=25, cycle_fail_after=25)
        threshold = "no_progress_fail_after"
        pattern = r"screen was unchanged after (\d+) consecutive steps"
    else:
        common.update(cycle_note_after=1, no_progress_note_after=25, no_progress_fail_after=25)
        threshold = "cycle_fail_after"
        pattern = r"been on this screen (\d+) times"

    hints = await HistoryBlock(config=AgentConfig(**common)).render(state)
    match = re.search(pattern, hints)
    note_count = int(match[1]) if match else 0
    guard_count = max(
        (
            value
            for value in range(1, len(state.actions) + 2)
            if check(state, AgentConfig(**common, **{threshold: value})) == Stop(reason)
        ),
        default=0,
    )
    assert note_count == guard_count == expected


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("screens", "calls_per_step", "note"),
    [
        ("AAAA", (3, 1, 1), "unchanged after 3 consecutive steps"),
        ("ABABABA", (2, 1, 2, 1, 1, 1), "been on this screen 3 times"),
    ],
)
async def test_default_progress_note_precedes_stop(screens, calls_per_step, note):
    state = _history(screens, calls_per_step)
    config = AgentConfig()
    assert check(state, config) is None
    assert note in await HistoryBlock(config=config).render(state)

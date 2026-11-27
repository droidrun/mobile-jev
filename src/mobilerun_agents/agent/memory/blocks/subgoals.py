import re

from llama_index.core.utils import get_tokenizer

from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.state import PlanningState, SubgoalResult
from mobilerun_agents.config.prompts import render_prompt

from . import StateBlock, clip


def _render_failure(result: SubgoalResult, redact: Redactor) -> str:
    summary = result.summary
    if summary is None:
        return ""
    blockers = list(summary.blockers)
    for thought in summary.thoughts:
        # Splitting a known protected value could send a fragment to the planner.
        parts = (
            [thought]
            if redact.masked_spans(thought)
            else [part.strip() for part in re.split(r"\n|(?<=[.!?])\s+", thought)]
        )
        blockers.extend(
            part
            for part in parts
            if re.search(
                r"\b(?:block\w*|can't|cannot|requires?|denied|unavailable|hides?)\b", part, re.I
            )
        )
    visits: dict[tuple[str, str], int] = {}
    for label, text, count in summary.screens:
        key = redact.text(label), redact.text(
            clip(text, min(120, len(text.partition("\n")[0])), redact)
        )
        visits[key] = visits.get(key, 0) + count
    lines = [f"Failure reason: {result.reason}"]
    for title, items in (
        ("Observed blockers", list(dict.fromkeys(blockers))[-3:]),
        ("Recent executor thoughts (observed data)", summary.thoughts[-3:]),
        (
            "Screens visited",
            [f"{label} — {text} ×{count}" for (label, text), count in visits.items()],
        ),
        ("Recent action results", summary.actions),
    ):
        if items:
            lines.extend(
                [f"{title}:", *(f"- {redact.text(clip(item, 240, redact))}" for item in items)]
            )
    return "\n".join(lines)


def _status(result: SubgoalResult) -> str:
    if result.success:
        return "Succeeded"
    return "Unfinished" if result.reason == "abstained" else "Failed"


class LastSubgoalBlock(StateBlock[PlanningState, str]):
    name: str = "last_subgoal"
    priority: int = 0

    async def render(self, state: PlanningState, redact: Redactor) -> str:
        if not state.results:
            return ""
        result = state.results[-1]
        return render_prompt(
            "assets/prompts/memory/last_subgoal.jinja2",
            result=result,
            status=_status(result),
            before=redact.value(result.before_screen),
            after=redact.value(result.after_screen),
            summary=_render_failure(result, redact),
            plan=state.plan,
        )


class SubgoalHistoryBlock(StateBlock[PlanningState, str]):
    name: str = "subgoal_history"
    priority: int = 1
    history: int

    async def render(self, state: PlanningState, redact: Redactor) -> str:
        results = state.results[-self.history - 1 : -1]
        if not results:
            return ""
        entries = [
            " ".join(
                f"{_status(result)}: {result.subgoal}{': ' + result.message if result.message else ''}".split()
            )
            for result in results
        ]
        return render_prompt("assets/prompts/memory/subgoals.jinja2", entries=entries)

    async def atruncate(self, content: str | list[object], tokens_to_truncate: int) -> str:
        if not content:
            return ""
        assert type(content) is str
        if tokens_to_truncate <= 0:
            return content
        tokenize = get_tokenizer()
        limit = len(tokenize(content)) - tokens_to_truncate
        heading, *entries = content.splitlines()
        for index in range(1, len(entries)):
            retained = "\n".join([heading, *entries[index:]])
            if len(tokenize(retained)) <= limit:
                return retained
        return ""

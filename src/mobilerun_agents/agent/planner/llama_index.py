"""Ask one LLM to plan the next subgoals or complete the overall run."""

import re
from copy import copy
from typing import Annotated

from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.tools import FunctionTool
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from workflows import Context

from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse, achat
from mobilerun_agents.agent.action_selector.llm.factory import adapter_capabilities
from mobilerun_agents.agent.action_selector.llm.parsing import parse_native_tool_calls
from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.workflows.core.state import PlanningState, SubgoalResult
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig

from .base import Finish, Plan, Planner, remaining_child_steps


class _SetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoals: Annotated[
        list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]],
        Field(min_length=1),
    ]
    findings: list[str] = Field(default_factory=list)


class _Complete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


def _decision(response: ChatResponse, error: str, last_turn: bool) -> Plan | Finish:
    parsed = parse_native_tool_calls(response.message)
    if len(parsed.calls) == 1:
        call = parsed.calls[0]
        if call.error:
            raise InvalidResponse(call.error)
        try:
            match call.name:
                case "set_plan" if not last_turn:
                    plan = _SetPlan.model_validate(call.parameters, strict=True)
                    return Plan(tuple(plan.subgoals), findings=tuple(plan.findings))
                case "complete":
                    answer = _Complete.model_validate(call.parameters, strict=True)
                    if last_turn and not answer.success and not answer.message.strip():
                        raise InvalidResponse(
                            "Explain what blocked the goal in the complete message."
                        )
                    return Finish(
                        answer.success, answer.message, answer.output if answer.success else None
                    )
        except ValidationError:
            raise InvalidResponse(error) from None
    raise InvalidResponse(error)


def _clip(text: str, limit: int, redact: Redactor) -> str:
    if 0 < limit < len(text):
        for start, end in redact.masked_spans(text):
            if start < limit < end:
                return text[:start]
    return text[:limit]


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
        key = label, _clip(text, min(120, len(text.partition("\n")[0])), redact)
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
            lines.extend([f"{title}:", *(f"- {_clip(item, 240, redact)}" for item in items)])
    return "\n".join(lines)


class LlamaIndexPlanner(Planner):
    def __init__(self, llm: LLM, config: AgentConfig | None = None) -> None:
        self.llm = llm
        self.config = config or AgentConfig()
        self._output_schema: dict[str, JSONValue] | None = None
        self._repair: str | None = None
        self._redact = Redactor()

    def bind(
        self,
        output_schema: dict[str, JSONValue] | None,
        repair: str | None,
        redact: Redactor,
    ) -> "LlamaIndexPlanner":
        planner = copy(self)
        planner._output_schema = output_schema
        planner._repair = repair
        planner._redact = redact
        return planner

    async def plan(self, ctx: Context[PlanningState]) -> Plan | Finish:
        state = await ctx.store.get_state()
        last_turn = remaining_child_steps(state) <= 0
        ui = state.current.ui if state.current else None
        message = ChatMessage(
            role="user",
            content=render_prompt(
                "assets/prompts/planning/turn.jinja2",
                state=state,
                repair=self._repair,
                last_turn=last_turn,
                summary=_render_failure(state.results[-1], self._redact) if state.results else "",
                package=ui.phone_state.get("packageName") if ui else None,
                activity=ui.phone_state.get("activityName") if ui else None,
                ui=ui.formatted_text if ui else "",
            ),
        )
        if state.current and state.current.screenshot and adapter_capabilities(self.llm).vision:
            message.blocks.append(state.current.screenshot)
        tools: list[FunctionTool] = []
        if not last_turn:
            tools.append(
                FunctionTool.from_defaults(
                    _SetPlan,
                    name="set_plan",
                    fn_schema=_SetPlan,
                    description=render_prompt("assets/prompts/planning/set_plan.jinja2"),
                )
            )
        tools.append(
            FunctionTool.from_defaults(
                _Complete,
                name="complete",
                fn_schema=_Complete,
                description=render_prompt(
                    "assets/prompts/complete/description.jinja2", output_schema=self._output_schema
                ),
            )
        )
        config = self.config.decision
        error = render_prompt(config.invalid_arguments_prompt)
        messages = [message]
        for attempt in range(config.max_response_repair_attempts):
            try:
                response = await achat(
                    self.llm,
                    messages,
                    timeout=config.timeout,
                    stream=self.config.streaming,
                    tools=tools,
                    allow_parallel_tool_calls=False,
                    tool_required=True,
                )
                return _decision(response, error, last_turn)
            except InvalidResponse as exc:
                if attempt + 1 == config.max_response_repair_attempts:
                    raise
                messages = [
                    message,
                    ChatMessage(
                        role="user",
                        content=render_prompt(
                            "assets/prompts/providers/repair.jinja2", error=str(exc)
                        ),
                    ),
                ]
        raise AssertionError("Planner repair budget must be positive")

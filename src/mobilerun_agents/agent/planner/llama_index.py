"""Ask one LLM to plan the next subgoals or complete the overall run."""

from copy import copy
from typing import Annotated

from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.memory import Memory
from llama_index.core.tools import BaseTool, FunctionTool, ToolSelection
from mobilerun.models.context import JSONValue
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from workflows import Context

from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse, achat
from mobilerun_agents.agent.action_selector.llm.factory import adapter_capabilities
from mobilerun_agents.agent.action_selector.llm.parsing import parse_native_tool_calls
from mobilerun_agents.agent.memory import read_memory
from mobilerun_agents.agent.workflows.core.state import PlanningState
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import AgentConfig

from .base import Finish, Plan, Planner, remaining_child_steps, remaining_steps


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


def _decision(
    response: ChatResponse, error: str, can_plan: bool, complete_only: bool, tool_names: set[str]
) -> Plan | Finish | ToolSelection:
    parsed = parse_native_tool_calls(response.message)
    if len(parsed.calls) == 1:
        call = parsed.calls[0]
        if call.error:
            raise InvalidResponse(call.error)
        try:
            match call.name:
                case "set_plan" if can_plan:
                    plan = _SetPlan.model_validate(call.parameters, strict=True)
                    return Plan(tuple(plan.subgoals), findings=tuple(plan.findings))
                case "complete":
                    answer = _Complete.model_validate(call.parameters, strict=True)
                    if complete_only and not answer.success and not answer.message.strip():
                        raise InvalidResponse(
                            "Explain what blocked the goal in the complete message."
                        )
                    return Finish(
                        answer.success, answer.message, answer.output if answer.success else None
                    )
                case name if name in tool_names:
                    return ToolSelection(
                        tool_id=call.call_id or name, tool_name=name, tool_kwargs=call.parameters
                    )
        except ValidationError:
            raise InvalidResponse(error) from None
    raise InvalidResponse(error)


class LlamaIndexPlanner(Planner):
    def __init__(self, llm: LLM, config: AgentConfig | None = None) -> None:
        self.llm = llm
        self.config = config or AgentConfig()
        self._output_schema: dict[str, JSONValue] | None = None
        self._repair: str | None = None
        self._tools: tuple[BaseTool, ...] = ()
        self._feedback: str | None = None

    def with_tools(self, tools: tuple[BaseTool, ...], feedback: str | None) -> "LlamaIndexPlanner":
        planner = copy(self)
        planner._tools = tools
        planner._feedback = feedback
        return planner

    def bind(
        self,
        output_schema: dict[str, JSONValue] | None,
        repair: str | None,
    ) -> "LlamaIndexPlanner":
        planner = copy(self)
        planner._output_schema = output_schema
        planner._repair = repair
        return planner

    async def plan(
        self, ctx: Context[PlanningState], memory: Memory
    ) -> Plan | Finish | ToolSelection:
        state = await ctx.store.get_state()
        can_plan = remaining_child_steps(state) > 0
        memory_tools = self._tools if remaining_steps(state) > 0 else ()
        complete_only = not can_plan and not memory_tools
        current = state.current
        ui = current.ui if current else None
        prompt = "assets/prompts/planning/turn.jinja2"
        system = ChatMessage(
            role="system",
            content=render_prompt(prompt, part="system", can_plan=can_plan),
        )
        step = render_prompt(
            prompt,
            part="step",
            repair=self._repair,
            tool_feedback=self._feedback,
            complete_only=complete_only,
            can_plan=can_plan,
            package=ui.phone_state.get("packageName") if ui else None,
            activity=ui.phone_state.get("activityName") if ui else None,
            ui=ui.formatted_text if ui else "",
            device_context=(
                current.model_dump(mode="json", include={"device_context"})["device_context"]
                if current
                else []
            ),
        )
        screenshot = current.screenshot if current else None
        if screenshot is not None and not adapter_capabilities(self.llm).vision:
            screenshot = None
        tools: list[BaseTool] = []
        tool_names = {member.metadata.get_name() for member in memory_tools}
        if "set_plan" in tool_names:
            raise ValueError("Duplicate tool name: set_plan")
        if can_plan:
            tools.append(
                FunctionTool.from_defaults(
                    _SetPlan,
                    name="set_plan",
                    fn_schema=_SetPlan,
                    description=render_prompt("assets/prompts/planning/set_plan.jinja2"),
                )
            )
        tools.extend(memory_tools)
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
        context = await read_memory(memory, ctx)
        if context and context[0].role == "system":
            # Native adapters expect a single system message; planner blocks are text-only.
            system.blocks.extend(context.pop(0).blocks)
        context_messages = [system, *context]
        content = step
        for attempt in range(config.max_response_repair_attempts):
            message = ChatMessage(role="user", content=content)
            if screenshot is not None:
                message.blocks.append(screenshot)
            try:
                response = await achat(
                    self.llm,
                    [*context_messages, message],
                    timeout=config.timeout,
                    stream=self.config.streaming,
                    tools=tools,
                    allow_parallel_tool_calls=False,
                    tool_required=True,
                )
                return _decision(response, error, can_plan, complete_only, tool_names)
            except InvalidResponse as exc:
                if attempt + 1 == config.max_response_repair_attempts:
                    raise
                repair = render_prompt("assets/prompts/providers/repair.jinja2", error=str(exc))
                content = f"{step}\n\n{repair}"
        raise AssertionError("Planner repair budget must be positive")

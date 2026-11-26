"""Ask an LLM for the next subgoals or the final result, without running actions."""

from time import monotonic
from typing import Annotated

from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.llms import LLM, ChatMessage, ChatResponse
from llama_index.core.tools import FunctionTool
from mobilerun.models.context import JSONValue
from mobilerun.utils.images import image_dimensions
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, create_model

from mobilerun_agents.agent.action_selector.llm.calls import (
    InvalidResponse,
    Usage,
    achat,
    usage_from_response,
)
from mobilerun_agents.agent.action_selector.llm.factory import (
    adapter_capabilities,
    llm_diagnostic_name,
)
from mobilerun_agents.agent.action_selector.llm.parsing import parse_native_tool_calls
from mobilerun_agents.agent.logs.diagnostics import Detail, ModelCall, Retry, redact, sink
from mobilerun_agents.agent.logs.redaction import redact_run_text
from mobilerun_agents.config.prompts import render_prompt

from .base import Finish, Plan, Planner, PlanningInput


class _SetPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subgoals: Annotated[
        list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]],
        Field(min_length=1),
    ]


class _Complete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    message: str
    output: dict[str, JSONValue] | None = None


def _decision(response: ChatResponse, error: str) -> Plan | Finish:
    parsed = parse_native_tool_calls(response.message)
    if (diagnostic := sink.get()) and (thought := redact(redact_run_text(parsed.thought))[:290]):
        diagnostic(Detail(section="planning", text=f"reasoning: {thought}"))
    if len(parsed.calls) != 1:
        raise InvalidResponse(error)
    call = parsed.calls[0]
    if call.error:
        raise InvalidResponse(call.error)
    try:
        match call.name:
            case "set_plan":
                args = _SetPlan.model_validate(call.parameters, strict=True)
                return Plan(tuple(args.subgoals))
            case "complete":
                answer = _Complete.model_validate(call.parameters, strict=True)
                return Finish(
                    answer.success, answer.message, answer.output if answer.success else None
                )
    except ValidationError:
        raise InvalidResponse(error) from None
    raise InvalidResponse(error)


async def _request(
    llm: LLM,
    request: PlanningInput,
    messages: list[ChatMessage],
    tools: list[FunctionTool],
    image: tuple[int, int] | None,
) -> ChatResponse:
    diagnostic = sink.get()
    started = monotonic()
    usage = Usage(requests=0)
    try:
        response = await achat(
            llm,
            messages,
            timeout=request.config.decision.timeout,
            stream=request.config.streaming,
            tools=tools,
            allow_parallel_tool_calls=False,
            tool_required=True,
        )
        usage = usage_from_response(response)
        return response
    except InvalidResponse as exc:
        usage = exc.usage or usage
        raise
    finally:
        if diagnostic:
            diagnostic(
                ModelCall(
                    name=llm_diagnostic_name(llm),
                    image=image,
                    seconds=monotonic() - started,
                    inputs=usage.request_tokens,
                    outputs=usage.response_tokens,
                    reasoning=usage.reasoning_tokens,
                    total=usage.total_tokens,
                )
            )


class LlamaIndexPlanner(Planner):
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    @property
    def supports_images(self) -> bool:
        return adapter_capabilities(self.llm).vision

    async def plan(self, request: PlanningInput) -> Plan | Finish:
        config = request.config.decision
        completion = (
            create_model(
                "CompletionArguments",
                __base__=_Complete,
                output=(request.output | None, None),  # type: ignore[arg-type]  # Runtime output type.
            )
            if request.output
            else _Complete
        )
        tools = [
            FunctionTool.from_defaults(
                _SetPlan,
                name="set_plan",
                fn_schema=_SetPlan,
                description=render_prompt("assets/prompts/planning/set_plan.jinja2"),
            ),
            FunctionTool.from_defaults(
                completion,
                name="complete",
                fn_schema=completion,
                description=render_prompt(
                    "assets/prompts/complete/description.jinja2",
                    output_schema=request.output.model_json_schema() if request.output else None,
                ),
            ),
        ]
        messages = [
            ChatMessage(
                role="user",
                content=render_prompt(
                    "assets/prompts/planning/turn.jinja2",
                    request=request,
                ),
            )
        ]
        diagnostic = sink.get()
        image = None
        if request.screenshot and self.supports_images:
            messages[0].blocks.append(ImageBlock(image=request.screenshot))
            if diagnostic:
                image = image_dimensions(request.screenshot)
        error = render_prompt(config.invalid_arguments_prompt)
        for attempt in range(1, config.max_response_repair_attempts + 1):
            try:
                return _decision(await _request(self.llm, request, messages, tools, image), error)
            except InvalidResponse as exc:
                if attempt == config.max_response_repair_attempts:
                    raise
                if diagnostic:
                    diagnostic(Retry(attempt=attempt + 1, reason="Invalid planner tool call"))
                messages = [
                    messages[0],
                    ChatMessage(
                        role="user",
                        content=render_prompt(
                            "assets/prompts/providers/repair.jinja2",
                            error=str(exc),
                        ),
                    ),
                ]
        raise AssertionError("Planner repair budget must be positive")

"""Hand off explicit delegation, low-confidence decisions and Jev completions."""

import json
from copy import copy
from dataclasses import replace

from llama_index.core.llms.llm import LLM

from mobilerun_agents.agent.action_selector.base import (
    ActionSelection,
    ActionSelector,
    sum_usage,
)
from mobilerun_agents.agent.action_selector.decision_model.base import DecisionSelector
from mobilerun_agents.agent.action_selector.llm.parsing import (
    ToolCallParseResult,
    ToolCallParseStatus,
)
from mobilerun_agents.agent.logs.diagnostics import Detail, sink
from mobilerun_agents.agent.run import Run
from mobilerun_agents.agent.tools.base import RegistryTool
from mobilerun_agents.agent.tools.registry import ToolRegistry
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DelegationConfig


class DelegatingSelector(ActionSelector):
    def __init__(
        self,
        primary: ActionSelector,
        delegate: ActionSelector,
        min_confidence: float | None = DelegationConfig().min_confidence,
        config: DelegationConfig | None = None,
    ) -> None:
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.primary, self.delegate = primary, delegate
        self.config = config or DelegationConfig(min_confidence=min_confidence)
        self.min_confidence = self.config.min_confidence

    @property
    def shared_llm(self) -> LLM | None:
        return self.delegate.shared_llm

    async def select(self, run: Run) -> ActionSelection:
        selection = await self.primary.select(run)
        diagnostic = sink.get()
        unsure = not selection.abstained and self._unsure(selection)
        explicit = False
        completion_reason = ""
        completion_hint: bool | None = None
        complete = run.registry.by_name.get("complete")
        if not selection.abstained:
            actions = selection.actions
            explicit = any(a.name == "delegate" for a in actions)
            if explicit and (
                "delegate" not in run.registry.by_name or len(actions) != 1 or actions[0].parameters
            ):
                return self._invalid(
                    selection,
                    render_selector_prompt(self.config.invalid_prompt, kind="parameters"),
                )
            finished = next((a for a in actions if a.name == "complete"), None)
            if complete is not None and finished is not None:
                match self.primary, finished.parameters.get("success"):
                    case DecisionSelector(), bool() as success if self.primary is not self.delegate:
                        completion_hint = success
                        completion_reason = "completion message"
                if "output" in complete.metadata.get_parameters_dict().get("properties", {}):
                    completion_reason = "typed completion"
            if not unsure and not explicit and not completion_reason:
                if diagnostic:
                    reason = (
                        f"confidence {selection.confidence} >= {self.min_confidence}"
                        if selection.confidence is not None and self.min_confidence is not None
                        else "primary accepted"
                    )
                    diagnostic(
                        Detail(
                            section="selector", text=f"{self.primary.diagnostic_name} ({reason})"
                        )
                    )
                return selection
            if completion_reason and len(actions) != 1:
                return self._invalid(
                    selection,
                    render_selector_prompt(self.config.invalid_prompt, kind="parameters"),
                )
        if diagnostic:
            reason = (
                "explicit delegate"
                if explicit
                else (
                    f"confidence {selection.confidence} < {self.min_confidence}"
                    if unsure
                    else "primary abstained" if selection.abstained else completion_reason
                )
            )
            diagnostic(
                Detail(
                    section="selector",
                    text=f"{self.primary.diagnostic_name} → {self.delegate.diagnostic_name} ({reason})",
                )
            )
        match complete, completion_hint:
            case RegistryTool(), bool() as success:
                run = self._with_completion_hint(run, complete, success)
        delegated = await self.delegate.select(run)
        if any(a.name == "delegate" for a in delegated.actions):
            return self._invalid(
                delegated,
                render_selector_prompt(self.config.invalid_prompt, kind="recursive"),
            )
        if delegated.abstained:
            if diagnostic and unsure and not explicit:
                diagnostic(
                    Detail(
                        section="selector",
                        text=f"{self.primary.diagnostic_name} (delegate abstained)",
                    )
                )
            return selection if unsure and not explicit else delegated
        if selection.abstained:
            return delegated
        return replace(
            delegated,
            usage=sum_usage(selection.usage, delegated.usage),
            memory_update="\n".join(
                filter(
                    None,
                    (
                        selection.memory_update,
                        delegated.memory_update,
                    ),
                )
            ),
        )

    @staticmethod
    def _with_completion_hint(run: Run, complete: RegistryTool, success: bool) -> Run:
        # Tool descriptions reach both native and XML delegates. This hint is not run history.
        hint = json.dumps({"primary_completion_hint": {"success": success}})
        metadata = replace(
            complete.metadata, description=f"{complete.metadata.description}\n{hint}"
        )

        async def invoke(**args: object) -> object:
            return (await complete.acall(**args)).raw_output

        hinted = RegistryTool(metadata, invoke, complete.action)
        delegate_run = copy(run)
        delegate_run.registry = ToolRegistry(
            tuple(hinted if tool is complete else tool for tool in run.registry)
        )
        return delegate_run

    def _unsure(self, selection: ActionSelection) -> bool:
        return (
            self.min_confidence is not None
            and selection.confidence is not None
            and selection.confidence < self.min_confidence
        )

    @staticmethod
    def _invalid(selection: ActionSelection, error: str) -> ActionSelection:
        return replace(
            selection,
            actions=[],
            failure=error,
            response=None if not selection.textual_arguments else selection.response,
            parse_result=ToolCallParseResult(
                thought=selection.thought,
                calls=[],
                status=ToolCallParseStatus.MALFORMED,
            ),
        )

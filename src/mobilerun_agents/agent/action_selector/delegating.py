"""Hand off explicit delegation, low-confidence decisions and Jev completions."""

import json
from copy import copy
from dataclasses import dataclass, field, replace

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
from mobilerun_agents.config.prompts import render_selector_prompt
from mobilerun_agents.config.user_config import DelegationConfig


@dataclass
class DelegationTurn:
    """Routing retained by the workflow for one observed step, including repairs."""

    delegated: bool = False
    completion_hint: bool | None = None
    previous_delegated: bool = False
    intent: str = field(default="", repr=False)

    def accept(self, selection: ActionSelection) -> None:
        self.previous_delegated = self.delegated
        if self.delegated and selection.thought.strip():
            self.intent = selection.thought.strip()

    def reset_attempt(self) -> None:
        self.delegated = False
        self.completion_hint = None


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

    async def select(self, run: Run, routing: DelegationTurn | None = None) -> ActionSelection:
        routing = routing if routing is not None else DelegationTurn()
        if routing.delegated or run.avoid_signature:
            routing.delegated = True
            return await self._select_delegate(run, routing.completion_hint)
        match self.primary:
            case DecisionSelector() as primary:
                selection = await primary.select(run, intent=routing.intent)
            case primary:
                selection = await primary.select(run)
        diagnostic = sink.get()
        confidence = selection.confidence
        minimum = self.min_confidence
        takeover_bar = (
            max(minimum, min(minimum + self.config.takeover_margin, 0.95))
            if routing.previous_delegated and minimum is not None
            else None
        )
        takeover_denied = takeover_bar is not None and (
            confidence is None or confidence < takeover_bar
        )
        unsure = not selection.abstained and (
            takeover_denied or (takeover_bar is None and self._unsure(selection))
        )
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
            if explicit:
                reason = "explicit delegate"
            elif selection.abstained:
                reason = "primary abstained"
            elif takeover_denied and takeover_bar is not None:
                reason = f"takeover bar {takeover_bar:g} not met: {confidence if confidence is not None else 'unknown'}"
            elif unsure:
                reason = f"confidence {confidence} < {minimum}"
            else:
                reason = completion_reason
            diagnostic(
                Detail(
                    section="selector",
                    text=f"{self.primary.diagnostic_name} → {self.delegate.diagnostic_name} ({reason})",
                )
            )
        routing.delegated = True
        routing.completion_hint = completion_hint
        delegated = await self._select_delegate(run, completion_hint)
        if delegated.abstained:
            if unsure and not explicit and not takeover_denied:
                routing.reset_attempt()
                if diagnostic:
                    diagnostic(
                        Detail(
                            section="selector",
                            text=f"{self.primary.diagnostic_name} (delegate abstained)",
                        )
                    )
                return selection
            return delegated
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

    async def _select_delegate(self, run: Run, completion_hint: bool | None) -> ActionSelection:
        match run.registry.by_name.get("complete"), completion_hint:
            case RegistryTool() as complete, bool() as success:
                run = self._with_completion_hint(run, complete, success)
        selection = await self.delegate.select(run)
        if any(call.name == "delegate" for call in selection.actions):
            return self._invalid(
                selection,
                render_selector_prompt(self.config.invalid_prompt, kind="recursive"),
                repairable=True,
            )
        return selection

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
        delegate_run.selection_include += (hinted,)
        return delegate_run

    def _unsure(self, selection: ActionSelection) -> bool:
        return (
            self.min_confidence is not None
            and selection.confidence is not None
            and selection.confidence < self.min_confidence
        )

    @staticmethod
    def _invalid(
        selection: ActionSelection, error: str, repairable: bool = False
    ) -> ActionSelection:
        return replace(
            selection,
            actions=[],
            abstained=False,
            failure=None if repairable else error,
            parse_error=error if repairable else None,
            response=None if not selection.textual_arguments else selection.response,
            parse_result=ToolCallParseResult(
                thought=selection.thought,
                calls=[],
                status=ToolCallParseStatus.MALFORMED,
            ),
        )

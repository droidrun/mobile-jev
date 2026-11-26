"""Try another selector when selection fails or cannot be executed."""

from collections.abc import Sequence
from dataclasses import replace

from mobilerun_agents.agent.action_selector.base import (
    ActionSelection,
    ActionSelector,
    sum_usage,
)
from mobilerun_agents.agent.run import Run
from mobilerun_agents.errors import SelectorCannotServe


def _usable(selection: ActionSelection | None, run: Run) -> bool:
    if selection is None or selection.failure is not None or selection.parse_error is not None:
        return False
    return any(
        a.error is None
        and a.name != "invalid"
        and (not run.registry.by_name or a.name in run.registry.by_name)
        for a in selection.actions
    )


class FallbackSelector(ActionSelector):
    def __init__(
        self,
        primary: ActionSelector,
        fallbacks: Sequence[ActionSelector],
        errors: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.primary, self.fallbacks, self.errors = primary, tuple(fallbacks), errors

    @property
    def selectors(self) -> tuple[ActionSelector, ...]:
        return (self.primary, *self.fallbacks)

    async def select(self, run: Run) -> ActionSelection:
        spent = []
        for selector in self.selectors[:-1]:
            try:
                selection = await selector.select(run)
            except SelectorCannotServe:
                raise
            except self.errors:
                continue
            if not selection.abstained:
                spent.append(selection.usage)
                if _usable(selection, run):
                    return replace(selection, usage=sum_usage(*spent))
        last = self.selectors[-1]
        selection = await last.select(run)
        return (
            replace(selection, usage=sum_usage(*spent, selection.usage))
            if not selection.abstained
            else selection
        )

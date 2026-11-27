"""Stop conditions derived from the raw action history."""

from dataclasses import dataclass
from typing import Literal

from mobilerun_agents.config.user_config import AgentConfig

from .core.state import AgentState
from .core.targets import cycle_visits, repeated_target, unchanged_steps


@dataclass(frozen=True)
class Stop:
    reason: Literal["max_steps", "stuck", "cycle"]


def check(state: AgentState, config: AgentConfig) -> Stop | None:
    """Check limits, repeated targets, no progress, and repeated screen visits."""
    actions = state.actions
    if (actions[-1].step if actions else 0) >= state.max_steps:
        return Stop("max_steps")

    if unchanged_steps(actions) >= config.no_progress_fail_after:
        return Stop("stuck")
    if repeated_target(actions) >= config.stuck_fail_after:
        return Stop("stuck")

    current = state.current.fingerprint if state.current is not None else None
    return Stop("cycle") if cycle_visits(actions, current) >= config.cycle_fail_after else None

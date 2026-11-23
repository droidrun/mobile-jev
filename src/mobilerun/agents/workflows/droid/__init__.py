"""Public agent types, deferred so events/config can be imported independently."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agents.workflows.droid.droid_agent import MobileAgent
    from mobilerun.agents.workflows.droid.state import MobileAgentState

__all__ = ["MobileAgent", "MobileAgentState"]


def __getattr__(name: str) -> type:
    # lazy: importing events must not initialize the agent loop.
    if name == "MobileAgent":
        from mobilerun.agents.workflows.droid.droid_agent import MobileAgent

        return MobileAgent
    if name == "MobileAgentState":
        from mobilerun.agents.workflows.droid.state import MobileAgentState

        return MobileAgentState
    raise AttributeError(name)

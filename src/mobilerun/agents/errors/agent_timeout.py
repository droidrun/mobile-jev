"""An agent operation exceeded its deadline."""

from mobilerun.errors.base import AgentError


class AgentTimeout(AgentError, TimeoutError):
    pass

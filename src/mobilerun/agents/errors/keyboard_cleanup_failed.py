from mobilerun.errors.base import AgentError


class KeyboardCleanupFailed(AgentError, RuntimeError):
    """Disabling the Mobilerun keyboard failed."""

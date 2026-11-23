"""Agent implementations and telemetry."""

from importlib import import_module
from typing import TYPE_CHECKING

from mobilerun.agents.logs import setup as _logging_setup  # noqa: F401

if TYPE_CHECKING:
    from mobilerun.agents.config.user_config import (
        AgentConfig,
        AppCardConfig,
        CredentialsConfig,
        ExecutorConfig,
        FastAgentConfig,
        LLMProfile,
        ManagerConfig,
        MobileConfig,
        ToolsConfig,
    )
    from mobilerun.agents.context.context_provider import ContextProvider, StepContext
    from mobilerun.agents.context.providers.app_card_context_provider import (
        AppCardContextProvider,
    )
    from mobilerun.agents.llm.llm_picker import load_llm
    from mobilerun.agents.operators.agent import AgentOperator
    from mobilerun.agents.operators.agent_macro import AgentMacroOperator
    from mobilerun.agents.operators.delegation import DelegationOperator
    from mobilerun.agents.workflows.droid import MobileAgent, MobileAgentState
    from mobilerun.agents.workflows.droid.events import ResultEvent

_EXPORTS = {
    "ContextProvider": ("mobilerun.agents.context.context_provider", "ContextProvider"),
    "StepContext": ("mobilerun.agents.context.context_provider", "StepContext"),
    "AppCardContextProvider": (
        "mobilerun.agents.context.providers.app_card_context_provider",
        "AppCardContextProvider",
    ),
    "DelegationOperator": ("mobilerun.agents.operators.delegation", "DelegationOperator"),
    "AgentOperator": ("mobilerun.agents.operators.agent", "AgentOperator"),
    "MobileAgent": ("mobilerun.agents.workflows.droid", "MobileAgent"),
    "MobileAgentState": ("mobilerun.agents.workflows.droid", "MobileAgentState"),
    "ResultEvent": ("mobilerun.agents.workflows.droid.events", "ResultEvent"),
    "load_llm": ("mobilerun.agents.llm.llm_picker", "load_llm"),
    "AgentMacroOperator": ("mobilerun.agents.operators.agent_macro", "AgentMacroOperator"),
    "MobileConfig": ("mobilerun.agents.config.user_config", "MobileConfig"),
    "AgentConfig": ("mobilerun.agents.config.user_config", "AgentConfig"),
    "FastAgentConfig": ("mobilerun.agents.config.user_config", "FastAgentConfig"),
    "ManagerConfig": ("mobilerun.agents.config.user_config", "ManagerConfig"),
    "ExecutorConfig": ("mobilerun.agents.config.user_config", "ExecutorConfig"),
    "AppCardConfig": ("mobilerun.agents.config.user_config", "AppCardConfig"),
    "ToolsConfig": ("mobilerun.agents.config.user_config", "ToolsConfig"),
    "CredentialsConfig": ("mobilerun.agents.config.user_config", "CredentialsConfig"),
    "LLMProfile": ("mobilerun.agents.config.user_config", "LLMProfile"),
}
__all__ = [
    "ContextProvider",
    "StepContext",
    "AppCardContextProvider",
    "AgentOperator",
    "DelegationOperator",
    "MobileAgent",
    "MobileAgentState",
    "ResultEvent",
    "load_llm",
    "AgentMacroOperator",
    "MobileConfig",
    "AgentConfig",
    "FastAgentConfig",
    "ManagerConfig",
    "ExecutorConfig",
    "AppCardConfig",
    "ToolsConfig",
    "CredentialsConfig",
    "LLMProfile",
]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        # Public-package seam: keep config/providers independent of the agent loop.
        module, attribute = _EXPORTS[name]
        return getattr(import_module(module), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

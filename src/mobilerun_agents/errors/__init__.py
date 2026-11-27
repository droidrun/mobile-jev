"""Error reporting for agent runs."""

from importlib.metadata import version

from mobilerun.errors import AgentError


class SelectorCannotServe(Exception):
    """The selector cannot choose a usable action from the offered tools."""


class UnknownModel(AgentError):
    """The installed adapter's local model registry does not recognise the configured model."""

    def __init__(self, provider: str, model: str) -> None:
        package = f"llama-index-llms-{provider}"
        alternative = (
            "use provider openai_like with a custom endpoint"
            if provider == "openai"
            else "choose a model it supports"
        )
        super().__init__(
            f"{provider} adapter ({package} {version(package)}) doesn't know model {model!r}. "
            f"Upgrade it (scripts/reinstall.sh) or {alternative}."
        )

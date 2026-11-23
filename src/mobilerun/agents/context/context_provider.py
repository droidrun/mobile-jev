"""Context supplied to agents during a task."""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from mobilerun.config.prompts import PromptLoader

if TYPE_CHECKING:
    from mobilerun.agents.workflows.droid.state import MobileAgentState


@dataclass
class StepContext:
    package_name: str
    activity_name: str
    instruction: str
    platform: str | None
    step: int
    ui_elements: list
    last_actions: list[dict]
    screenshot: bytes | None


class ContextProvider(ABC):
    @abstractmethod
    async def get_context(self, context: StepContext) -> str:
        """Return additional instructions for the current step."""


class DisabledContextProvider(ContextProvider):
    async def get_context(self, context: StepContext) -> str:
        return ""


async def render_extra_context(extra_context: str) -> str:
    path = Path(__file__).resolve().parents[1] / "assets/prompts/fast_agent/extra_context.jinja2"
    return await PromptLoader.load_prompt(str(path), {"extra_context": extra_context})


async def update_context(
    provider: ContextProvider, state: "MobileAgentState", screenshot: bytes | str | None
) -> None:
    """Refresh the shared context after reading the current screen."""
    if isinstance(screenshot, str):
        try:
            screenshot = base64.b64decode(screenshot, validate=True)
        except ValueError:
            screenshot = None
    actions = [
        {"action": action, "summary": summary}
        for action, summary in zip(
            state.action_history[-5:], state.summary_history[-5:], strict=False
        )
    ]
    context = StepContext(
        package_name=state.current_package_name,
        activity_name=state.current_activity_name,
        instruction=state.instruction,
        platform=state.platform,
        step=state.step_number,
        ui_elements=state.a11y_tree,
        last_actions=actions,
        screenshot=screenshot,
    )
    try:
        state.extra_context = await provider.get_context(context)
    except (OSError, UnicodeError, httpx.RequestError) as exc:
        logging.getLogger("mobilerun").warning("Error loading context: %s", exc)
        state.extra_context = ""

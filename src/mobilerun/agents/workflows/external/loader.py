"""External agent loader — dynamic imports.

External agents are self-contained modules that receive raw ADB access
via ``async_adbutils.AdbDevice``. They bring their own LLM client, prompts,
parsing, and action loop — zero imports from ``mobilerun``.

An external agent can be either:
- A single file: ``mobilerun/agents/workflows/external/my_agent.py``
- A package:     ``mobilerun/agents/workflows/external/my_agent/__init__.py``

Required contract::

    from async_adbutils import AdbDevice

    async def run(
        device: AdbDevice,       # raw ADB, already connected
        instruction: str,        # the task
        config: dict,            # from external_agents.<name> in config.yaml
        max_steps: int,          # step limit
    ) -> dict:                   # {"success": bool, "reason": str, "steps": int}

Optional: ``DEFAULT_CONFIG: dict`` — merged under the user's config.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

from mobilerun.agents.context.context_provider import (
    ContextProvider,
    render_extra_context,
    update_context,
)

if TYPE_CHECKING:
    from mobilerun.agents.workflows.droid.state import MobileAgentState

logger = logging.getLogger("mobilerun")

_EXTERNAL_DIR = Path(__file__).parent


class ExternalAgentModule(TypedDict):
    """Type for a loaded external agent module."""

    run: Callable
    config: Dict[str, Any]


async def run_with_context(
    run: Callable[..., Awaitable[dict[str, Any]]],
    provider: ContextProvider | None,
    state: MobileAgentState,
    **kwargs: Any,
) -> dict[str, Any]:
    """Supply startup context; the external run contract has no step hook."""
    instruction = state.instruction
    if provider is not None:
        await update_context(provider, state, None)
        if state.extra_context:
            instruction += "\n\n" + await render_extra_context(state.extra_context)
    return await run(instruction=instruction, **kwargs)


def list_agents() -> List[str]:
    """Discover available external agents by scanning the external/ directory.

    Returns:
        Sorted list of agent names (module stems or package directory names).
    """
    agents: list[str] = []
    for item in _EXTERNAL_DIR.iterdir():
        if item.name == "loader.py" or item.name.startswith(("_", ".")):
            continue
        if item.is_file() and item.suffix == ".py":
            agents.append(item.stem)
        elif item.is_dir() and (item / "__init__.py").exists():
            agents.append(item.name)
    return sorted(agents)


def load_agent(name: str) -> Optional[ExternalAgentModule]:
    """Dynamically load an external agent by name.

    Args:
        name: Agent module name (e.g., ``"my_agent"``).

    Returns:
        Dict with ``run`` function and ``config`` defaults, or *None* on failure.
    """
    try:
        module = importlib.import_module(f"mobilerun.agents.workflows.external.{name}")

        # Third-party plugin modules may omit the required entry point.
        if not hasattr(module, "run"):
            logger.error(f"External agent '{name}' missing run() function")
            return None

        if not asyncio.iscoroutinefunction(module.run):
            logger.error(
                f"External agent '{name}' has a sync run() — it must be async (async def run)"
            )
            return None

        return {
            "run": module.run,
            # Plugin modules may omit their optional defaults.
            "config": getattr(module, "DEFAULT_CONFIG", {}),
        }

    except ImportError as e:
        logger.error(f"Failed to load external agent '{name}': {e}")
        return None

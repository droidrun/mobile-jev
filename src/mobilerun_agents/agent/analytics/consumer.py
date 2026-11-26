"""Allowlisted product events from a single workflow stream."""

import asyncio
import logging
from pathlib import Path
from threading import Lock
from typing import TypeVar
from uuid import UUID, uuid4

from filelock import FileLock
from mobilerun.config.user_config import TelemetryConfig
from posthog import Posthog

from mobilerun_agents.agent.workflows.core.result import AgentResult
from mobilerun_agents.config.settings import get_agent_settings

_log = logging.getLogger(__name__)
_posthog: Posthog | None = None
_lock = Lock()
T = TypeVar("T")


def _client() -> Posthog:
    global _posthog
    with _lock:
        if _posthog is None:
            settings = get_agent_settings()
            _posthog = Posthog(
                settings.MOBILERUN_POSTHOG_PROJECT_API_KEY, host=settings.MOBILERUN_POSTHOG_HOST
            )
        return _posthog


def _identity() -> str:
    path = Path.home() / ".droidrun" / "user_id"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        try:
            return str(UUID(path.read_text().strip()))
        except (FileNotFoundError, ValueError):
            identity = str(uuid4())
            path.write_text(identity)
            return identity


class RunAnalytics:
    def __init__(self, config: TelemetryConfig, run_id: str) -> None:
        self._run_id = run_id
        self._client: Posthog | None = None
        self._packages: set[str] = set()
        self._activities: set[tuple[str, str]] = set()
        self._previous: tuple[str, str | None] | None = None
        self._finished = False
        if config.enabled and get_agent_settings().telemetry_enabled:
            try:
                self._identity = _identity()
                self._client = _client()
                self._capture("run_started")
            except (OSError, ValueError, RuntimeError):
                _log.warning("Unable to initialize analytics")

    def _capture(self, name: str, **properties: str | int | bool | None) -> None:
        if self._client is not None:
            try:
                self._client.capture(
                    name,
                    distinct_id=self._identity,
                    properties={"run_id": self._run_id, **properties},
                )
            except Exception:
                _log.warning("Unable to capture %s", name)

    def on_observe(self, package: str | None, activity: str | None, step: int) -> None:
        if self._finished or self._client is None:
            return
        if not package or (package, activity) == self._previous:
            return
        self._previous = package, activity
        self._packages.add(package)
        if activity:
            self._activities.add((package, activity))
        self._capture("app_visited", package_name=package, activity_name=activity, step_number=step)

    async def finish(self, result: AgentResult[T]) -> None:
        if self._finished or self._client is None:
            return
        self._finished = True
        self._capture(
            "run_finished",
            success=result.success,
            steps=result.steps,
            unique_packages_count=len(self._packages),
            unique_activities_count=len(self._activities),
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._client.flush),
                get_agent_settings().MOBILERUN_FINALIZE_TELEMETRY_SECONDS,
            )
        except Exception:
            _log.warning("Unable to flush analytics before deadline")

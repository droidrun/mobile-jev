from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable


class OAuthLoginDeadline:
    """One monotonic timeout budget shared by every OAuth login stage."""

    def __init__(
        self,
        timeout_seconds: float,
        *,
        timeout_message: str = "OAuth login timed out.",
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        _expires_at: float | None = None,
    ) -> None:
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OAuth login timeout must be a finite positive number.")
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._timeout_message = timeout_message
        self._expires_at = self._clock() + timeout if _expires_at is None else _expires_at

    @property
    def expires_at(self) -> float:
        return self._expires_at

    def remaining(self, *, cap: float | None = None) -> float:
        remaining = self._expires_at - self._clock()
        if remaining <= 0:
            raise TimeoutError(self._timeout_message)
        if cap is None:
            return remaining
        maximum = float(cap)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("OAuth request timeout cap must be finite and positive.")
        return min(remaining, maximum)

    def check(self) -> None:
        self.remaining()

    def sleep(self, delay_seconds: float) -> None:
        delay = max(0.0, float(delay_seconds))
        if delay:
            self._sleeper(min(delay, self.remaining()))
        self.check()

    def limited_to(self, timeout_seconds: float) -> OAuthLoginDeadline:
        """Return a view capped by a provider-issued expiry, never a reset budget."""
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OAuth login timeout cap must be finite and positive.")
        return OAuthLoginDeadline(
            timeout,
            timeout_message=self._timeout_message,
            clock=self._clock,
            sleeper=self._sleeper,
            _expires_at=min(self._expires_at, self._clock() + timeout),
        )


def open_browser_async(url: str, opener: Callable[[str], object]) -> None:
    """Launch a browser without letting OS integration consume the login budget."""

    def _open() -> None:
        try:
            opener(url)
        except Exception:
            # The URL is always printed, so browser integration is best-effort.
            return

    threading.Thread(target=_open, daemon=True).start()

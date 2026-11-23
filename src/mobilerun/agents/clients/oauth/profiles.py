"""Safe shared storage for Mobilerun authentication profiles.

All OAuth providers and saved API keys share one JSON object.  Writes therefore
need to be serialized across processes and must preserve slots owned by other
providers.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from filelock import FileLock
from mobilerun.config.env import secret_value
from mobilerun.errors.base import CredentialError

from mobilerun.agents.config.env import get_agent_secrets
from mobilerun.agents.config.settings import get_agent_settings


class AuthProfileFormatError(CredentialError, ValueError):
    """Raised when an existing auth-profiles file is not a JSON object."""


_T = TypeVar("_T")
# fchmod is unavailable on some supported platforms.
_FCHMOD: Callable[[int, int], None] | None = getattr(os, "fchmod", None)


class AuthProfileTransaction(AbstractContextManager["AuthProfileTransaction"]):
    """A locked read/modify/write transaction over an auth profile file."""

    def __init__(
        self,
        store: "AuthProfileStore",
        *,
        lock_timeout: float | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._lock = FileLock(str(store.lock_path))
        self._lock_timeout = lock_timeout
        self._before_commit = before_commit
        self._profile: dict[str, Any] = {}
        self._dirty = False

    def __enter__(self) -> "AuthProfileTransaction":
        self._store.path.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_timeout is None:
            self._lock.acquire()
        else:
            self._lock.acquire(timeout=self._lock_timeout)
        try:
            # The lock file has no secrets, but keeping it private prevents
            # other local users from deliberately interfering with writers.
            try:
                os.chmod(self._store.lock_path, 0o600)
            except OSError:
                pass
            self._profile = self._store._read_unlocked()
        except BaseException:
            self._lock.release()
            raise
        return self

    @property
    def profile(self) -> dict[str, Any]:
        """The locked profile object. Mutate it only via :meth:`update`."""
        return self._profile

    def get_slot(self, slot: str) -> dict[str, Any] | None:
        value = self._profile.get(slot)
        return dict(value) if isinstance(value, dict) else None

    def set_slot(self, slot: str, payload: dict[str, Any]) -> None:
        self._profile[slot] = dict(payload)
        self._dirty = True

    def update(self, updater: Callable[[dict[str, Any]], _T]) -> _T:
        result = updater(self._profile)
        self._dirty = True
        return result

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if exc_type is None and self._dirty:
                self._store._write_unlocked(
                    self._profile,
                    before_commit=self._before_commit,
                )
        finally:
            self._lock.release()
        return None


class AuthProfileStore:
    """Cross-process-safe JSON object store used by all auth providers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")

    def transaction(
        self,
        *,
        lock_timeout: float | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> AuthProfileTransaction:
        return AuthProfileTransaction(
            self,
            lock_timeout=lock_timeout,
            before_commit=before_commit,
        )

    def read_profile(self) -> dict[str, Any]:
        with self.transaction() as transaction:
            return dict(transaction.profile)

    def read_slot(self, slot: str) -> dict[str, Any] | None:
        with self.transaction() as transaction:
            return transaction.get_slot(slot)

    def update_slot(
        self,
        slot: str,
        payload: dict[str, Any],
        *,
        lock_timeout: float | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        with self.transaction(
            lock_timeout=lock_timeout,
            before_commit=before_commit,
        ) as transaction:
            transaction.set_slot(slot, payload)

    def update_profile(self, updater: Callable[[dict[str, Any]], _T]) -> _T:
        with self.transaction() as transaction:
            return transaction.update(updater)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthProfileFormatError(
                f"Cannot update malformed authentication profile {self.path}. "
                "Repair or remove the file, then retry."
            ) from exc
        if not isinstance(payload, dict):
            raise AuthProfileFormatError(
                f"Authentication profile {self.path} must contain a JSON object."
            )
        return payload

    def _write_unlocked(
        self,
        profile: dict[str, Any],
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd: int | None = None
        tmp_path: Path | None = None
        try:
            fd, raw_tmp_path = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(raw_tmp_path)
            if _FCHMOD is not None:
                _FCHMOD(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None
                json.dump(profile, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if before_commit is not None:
                before_commit()
            os.replace(tmp_path, self.path)
            tmp_path = None
            os.chmod(self.path, 0o600)

            # Persist the directory entry as well as the file contents.
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass


# API key lookup and persistence helpers.

API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

_API_KEYS_SECTION = "apiKeys"


@dataclass(frozen=True)
class ApiKeySources:
    """API key values from the shell and the saved env file."""

    shell: str = ""
    saved: str = ""


def _load_saved_api_keys() -> dict[str, str]:
    """Read the apiKeys section from the shared auth-profiles.json."""
    path = get_agent_settings().auth_profiles_path
    if not path.exists():
        return {}
    try:
        data = AuthProfileStore(path).read_profile()
        section = data.get(_API_KEYS_SECTION)
        if isinstance(section, dict):
            return {k: str(v) for k, v in section.items() if v}
        return {}
    except Exception:
        return {}


def load_env_key_sources() -> dict[str, ApiKeySources]:
    """Load API keys from shell env vars and the shared auth-profiles file.

    The returned mapping keeps the two sources separate so callers can decide
    whether to prefer the live shell environment or the persisted file.
    """
    saved_keys = _load_saved_api_keys()
    result: dict[str, ApiKeySources] = {}

    secrets = get_agent_secrets()
    shell_keys = {
        "google": secrets.GOOGLE_API_KEY,
        "gemini": secrets.GEMINI_API_KEY,
        "openai": secrets.OPENAI_API_KEY,
        "xai": secrets.XAI_API_KEY,
        "anthropic": secrets.ANTHROPIC_API_KEY,
        "zai": secrets.ZAI_API_KEY,
        "minimax": secrets.MINIMAX_API_KEY,
    }
    for slot, secret in shell_keys.items():
        result[slot] = ApiKeySources(
            shell=secret_value(secret) or "",
            saved=saved_keys.get(slot, ""),
        )

    return result


def load_env_keys() -> dict[str, str]:
    """Load API keys. The saved file takes precedence over shell env vars.

    Returns:
        Dict mapping slot name (e.g. "google") to key value.
    """
    result: dict[str, str] = {}
    for slot, sources in load_env_key_sources().items():
        result[slot] = sources.saved or sources.shell
    return result


def resolve_env_key(slot: str, source: str = "auto") -> str:
    """Resolve an API key for a provider slot from a specific source.

    Args:
        slot: Provider slot name, e.g. "openai" or "anthropic".
        source: "auto" (saved file first, then shell), "env" (shell only), or
            "file" (saved env file only).
    """
    sources = load_env_key_sources().get(slot, ApiKeySources())
    if source == "env":
        return sources.shell
    if source == "file":
        return sources.saved
    return sources.saved or sources.shell


def save_env_keys(keys: dict[str, str]) -> None:
    """Persist API keys to the apiKeys section of auth-profiles.json.

    Also sets them as env vars in the current process.

    Args:
        keys: Dict mapping slot name (e.g. "google") to key value.
    """
    known_keys = {slot: val for slot, val in keys.items() if slot in API_KEY_ENV_VARS}

    def _update(existing: dict) -> None:
        api_keys = existing.get(_API_KEYS_SECTION, {})
        if not isinstance(api_keys, dict):
            api_keys = {}
        for slot, val in known_keys.items():
            if val:
                api_keys[slot] = val
            else:
                api_keys.pop(slot, None)
        existing[_API_KEYS_SECTION] = api_keys

    # Persist first. In particular, a malformed file must not be replaced or
    # leave the current process believing a key was saved successfully.
    AuthProfileStore(get_agent_settings().auth_profiles_path).update_profile(_update)
    for slot, val in known_keys.items():
        env_var = API_KEY_ENV_VARS[slot]
        if val:
            os.environ[env_var] = val  # Persist the chosen provider key for SDKs.
        else:
            os.environ.pop(env_var, None)  # Remove the chosen provider key from SDKs.

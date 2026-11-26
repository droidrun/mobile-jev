"""Private, atomically replaced records under the mobilerun config directory."""

import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from mobilerun.config.settings import get_settings
from mobilerun.errors import CredentialFormatError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_PROVIDER = re.compile(r"[a-z][a-z0-9_]{0,63}")


def provider_name(provider: str) -> str:
    name = provider.lower()
    if not _PROVIDER.fullmatch(name):
        raise ValueError("Invalid credential provider name.")
    return name


class Saved(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    api_key: str = Field(min_length=1, repr=False)


def _folder() -> Path:
    return get_settings().MOBILERUN_USER_CONFIG_DIR / "agent" / "auth"


def _path(provider: str) -> Path:
    return _folder() / (provider_name(provider) + ".json")


def list_providers() -> list[str]:
    return sorted(
        p.stem for p in _folder().glob("*.json") if _PROVIDER.fullmatch(p.stem) and p.is_file()
    )


def delete(provider: str) -> None:
    _path(provider).unlink(missing_ok=True)


def load(provider: str) -> Saved | None:
    try:
        return Saved.model_validate_json(_path(provider).read_bytes())
    except FileNotFoundError:
        return None
    except ValidationError:
        raise CredentialFormatError("Invalid saved agent credential; save an API key.") from None


def save_key(provider: str, api_key: str) -> None:
    record = Saved(api_key=api_key)
    path = _path(provider)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    stream = NamedTemporaryFile(dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            stream.write(record.model_dump_json(exclude_none=True).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

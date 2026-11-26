"""Locate a replay target by run ID or explicit path."""

from os import PathLike, fspath
from pathlib import Path

from mobilerun.macro.settings import get_macro_settings


def resolve_macro_target(target: str | PathLike[str]) -> Path:
    path = Path(target).expanduser()
    if path.name != fspath(target) or path.name in {"", "..", "macro.json"}:
        return path

    saved = Path(get_macro_settings().macro_directory).expanduser().absolute() / path.name
    if not saved.exists() and (path.is_file() or (path / "macro.json").is_file()):
        return path
    return saved

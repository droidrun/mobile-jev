"""Identify the macro and version behind a resolved source file."""

import re
from pathlib import Path


def macro_identity(source: Path) -> tuple[str, int]:
    """Derive the publication name and version floor from the source path."""
    if match := re.fullmatch(r"v([1-9][0-9]*)\.jsonl", source.name):
        return source.parent.name, int(match[1])
    return source.stem, 0

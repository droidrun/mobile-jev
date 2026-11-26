"""Resolve agent assets and absolute prompt overrides."""

from pathlib import Path


def resolve_prompt_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Prompt path not found: {resolved}")
    candidate = (Path(__file__).resolve().parents[1] / path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Prompt not found in package: {candidate}")

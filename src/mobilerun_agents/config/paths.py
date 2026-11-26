"""Resolve agent assets and working-directory overrides."""

from pathlib import Path


def resolve_prompt_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return path
    candidates = (
        Path.cwd() / path,
        Path(__file__).resolve().parents[1] / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Path not found in: " + ", ".join(map(str, candidates)))

"""Agent asset resolution, preserving working-directory and legacy paths."""

from pathlib import Path

from mobilerun.config.paths import PathResolver as BasePathResolver


class PathResolver(BasePathResolver):
    @staticmethod
    def get_project_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def resolve(
        path: str | Path, create_if_missing: bool = False, must_exist: bool = False
    ) -> Path:
        path = Path(path).expanduser()
        if path.is_absolute() or create_if_missing:
            return BasePathResolver.resolve(path, create_if_missing, must_exist)
        # Preserve both historical CWD spellings and package-root overrides.
        candidates = [
            Path.cwd() / path,
            BasePathResolver.get_project_root() / path,
            PathResolver.get_project_root() / path,
        ]
        if (
            len(path.parts) >= 2
            and path.parts[0] == "config"
            and path.parts[1] in {"prompts", "app_cards"}
        ):
            candidates.append(PathResolver.get_project_root() / "assets" / Path(*path.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if must_exist:
            raise FileNotFoundError("Path not found in: " + ", ".join(map(str, candidates)))
        return Path.cwd() / path

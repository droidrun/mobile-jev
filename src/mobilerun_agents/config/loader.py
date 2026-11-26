"""Load agent settings without rewriting the shared device configuration."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from mobilerun.config.settings import get_settings

from .user_config import MobileConfig

_override: ContextVar[MobileConfig | None] = ContextVar("agents_config", default=None)


class ConfigLoader:
    @staticmethod
    @contextmanager
    def override(config: MobileConfig) -> Iterator[None]:
        token = _override.set(config)
        try:
            yield
        finally:
            _override.reset(token)

    @staticmethod
    def get_user_config_path() -> Path:
        return get_settings().MOBILERUN_USER_CONFIG_DIR / "agents.yaml"

    @classmethod
    def load(cls, config_path: str | None = None) -> MobileConfig:
        scoped = _override.get()
        if config_path is None and scoped is not None:
            return scoped
        path = Path(config_path) if config_path else cls.get_user_config_path()
        return MobileConfig.from_yaml(path) if config_path or path.exists() else MobileConfig()

    @classmethod
    def save(cls, config: MobileConfig) -> Path:
        path = cls.get_user_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temp = Path(file.name)
            try:
                # exclude_unset would also discard edits made inside a default section.
                choices = config.model_dump(mode="json", exclude_defaults=True)
                MobileConfig.model_validate(choices)
                yaml.safe_dump(choices, file, sort_keys=False)  # type: ignore[no-untyped-call]
                file.close()
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        return path

"""Load agent settings without rewriting the shared device configuration."""

from typing import Any

import yaml
from mobilerun.config.loader import ConfigFile

from .settings import get_agent_settings
from .user_config import MobileConfig, RecordingConfig


class _ConfigYAML(yaml.SafeLoader):
    # Keep on/off as strings without changing true/false.
    yaml_implicit_resolvers = {
        initial: [
            (tag, pattern)
            for tag, pattern in rules
            if tag != "tag:yaml.org,2002:bool" or initial in "tTfF"
        ]
        for initial, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


class ConfigLoader(ConfigFile[MobileConfig]):
    CONFIG_FILE = "agents.yaml"
    config_type = MobileConfig

    @classmethod
    def load(cls, config_path: str | None = None) -> MobileConfig:
        if config_path is None and (resolved := cls._override.get()) is not None:
            return resolved
        config = super().load(config_path)
        enabled = get_agent_settings().MOBILERUN_AGENT_RECORDING_ENABLED
        if enabled is not None and enabled != config.recording.enabled:
            effective = config.model_copy(update={"recording": RecordingConfig(enabled=enabled)})
            effective._recording_override = (config.recording.enabled, enabled)
            return effective
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MobileConfig:
        return cls.config_type.validate_file_data(data)

    @classmethod
    def to_dict(cls, config: MobileConfig) -> dict[str, Any]:
        if (
            config._recording_override is not None
            and config.recording.enabled == config._recording_override[1]
        ):
            config = config.model_copy(
                update={"recording": RecordingConfig(enabled=config._recording_override[0])}
            )
        data = config.model_dump(mode="json", exclude_defaults=True)
        cls.config_type.model_validate(data)
        return data

    @classmethod
    def parse_yaml(cls, text: str) -> object:
        try:
            data = yaml.load(text, Loader=_ConfigYAML)  # type: ignore[no-untyped-call]
        except yaml.YAMLError:
            raise ValueError("Invalid agents.yaml syntax") from None
        if data is None:
            return {}
        if type(data) is not dict:
            # Preserve the Pydantic error for a non-mapping document root.
            cls.config_type.model_validate(data)
        return data

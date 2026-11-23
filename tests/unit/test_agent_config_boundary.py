"""Shared config persistence stays independent of typed optional sections."""

from copy import deepcopy

import yaml
from mobilerun.config.loader import ConfigLoader
from mobilerun.config.user_config import MobileConfig

from mobilerun.agents.config.loader import ConfigLoader as AgentConfigLoader
from mobilerun.agents.config.user_config import MobileConfig as AgentConfig


def test_mobile_agent_normalizes_shared_and_agent_configs():
    from llama_index.core.llms.mock import MockLLM

    from mobilerun.agents.workflows.droid.droid_agent import MobileAgent

    configs = (
        MobileConfig.from_dict({"agent": {"max_steps": 7}}),
        AgentConfig.from_dict({"agent": {"max_steps": 7}}),
    )
    for config in configs:
        agent = MobileAgent("goal", config=config, llms=MockLLM())
        assert agent.config.agent.max_steps == 7


def test_shared_config_round_trips_raw_optional_sections_without_mutating_input():
    data = {
        "_version": 8,
        "device": {"serial": "test"},
        "agent": {"max_steps": 3, "manager": {"stateless": True}},
        "llm_profiles": {"manager": {"provider": "GoogleGenAI", "model": "test"}},
        "mcp": {"enabled": True, "servers": {"local": {"command": "echo"}}},
        "credentials": {"enabled": True, "file_path": "credentials.yaml"},
        "tools": {"disabled_tools": []},
    }
    original = deepcopy(data)
    shared = MobileConfig.from_dict(data)
    for section in ("agent", "llm_profiles", "mcp", "credentials", "tools"):
        assert shared.to_dict()[section] == data[section]
    typed = AgentConfig.from_dict(shared.to_dict())
    assert typed.agent.max_steps == 3
    assert typed.agent.manager.stateless is True
    assert typed.mcp.servers["local"].command == "echo"
    assert typed.tools.disabled_tools == []
    assert data == original


def test_both_loaders_accept_and_migrate_the_same_old_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"_version": 1, "device": {"serial": "test"}}))
    shared = ConfigLoader.load(str(path))
    typed = AgentConfigLoader.load(str(path))
    assert shared.device.serial == typed.device.serial == "test"
    assert isinstance(shared.agent, dict)
    assert typed.agent.fast_agent.parallel_tools is True
    assert yaml.safe_load(path.read_text())["_version"] == 8


def test_both_loaders_use_config_directory_environment_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr("platformdirs.user_config_dir", lambda *args: str(tmp_path / "default"))
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    for loader in (ConfigLoader, AgentConfigLoader):
        directory = tmp_path / loader.__module__
        monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(directory))
        config = loader.config_type()
        config.device.serial = "test-device"
        assert loader.save(config) == directory / "config.yaml"
        assert loader.load().device.serial == "test-device"

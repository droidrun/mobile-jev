import asyncio

import pytest
from pydantic import ValidationError

from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import PROVIDER_KINDS, MobileConfig


def test_llm_profiles_validate_the_same_providers_the_loader_supports():
    assert {name for name, kind in PROVIDER_KINDS.items() if kind == "llm"} == MODELS.keys()


def test_clean_config_round_trip(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "agent: {max_steps: 7}\ncheckpoints: {folder: runs}\nmemory: {token_limit: 9000}\n"
    )
    config = ConfigLoader.load(str(path))
    assert config.agent.max_steps == 7 and config.memory.token_limit == 9000
    assert str(config.checkpoints.folder) == "runs"
    assert config.profiles == {} and config.setups == {} and config.default_setup is None
    assert MobileConfig.model_validate(config.model_dump(mode="json")) == config


def test_run_budgets():
    config = MobileConfig().agent
    assert config.max_steps == 25
    assert config.planning_max_steps == 40
    assert config.decision.timeout == 120
    assert config.workflow_timeout is None


def test_scoped_config_is_nested_and_task_local():
    first, second = MobileConfig(), MobileConfig()
    first.agent.max_steps, second.agent.max_steps = 7, 8

    async def task(config):
        with ConfigLoader.override(config):
            await asyncio.sleep(0)
            assert ConfigLoader.load() is config
            with ConfigLoader.override(MobileConfig()):
                assert ConfigLoader.load().agent.max_steps == 25
            assert ConfigLoader.load() is config

    async def both():
        await asyncio.gather(task(first), task(second))

    asyncio.run(both())
    assert ConfigLoader.load().agent.max_steps == 25


@pytest.mark.parametrize(
    "data",
    [
        {"telemetry": {}},
        {"agent": {"fast_agent": {}}},
        {"mcp": {}},
        {"agent": {"max_steps": 0}},
        {"llm_profiles": {"default": {"auth_mode": "oauth"}}},
        {"profiles": {"jev": {"provider": "typesafe", "min_confidence": 4}}},
        {"profiles": {"jev": {"provider": "typesafe", "model": "jev"}}},
    ],
)
def test_rejects_invalid_or_removed_settings(data):
    with pytest.raises(ValidationError):
        MobileConfig.model_validate(data)


def test_unquoted_vision_modes_in_yaml(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text(
        "profiles:\n"
        "  luna: {provider: openai, model: gpt-6-luna, vision: off}\n"
        "  jev: {provider: typesafe, vision: on}\n"
        "agent: {streaming: false}\n"
    )
    config = ConfigLoader.load(str(path))
    assert config.profiles["luna"].vision == "off"
    assert config.profiles["jev"].vision == "on"
    assert not config.agent.streaming


def test_asset_resolver_does_not_rewrite_old_config_paths(tmp_path, monkeypatch):
    from mobilerun_agents.config.paths import resolve_prompt_path

    monkeypatch.chdir(tmp_path)
    asset = "assets/prompts/planning/turn.jinja2"
    assert resolve_prompt_path(asset).is_file()
    with pytest.raises(FileNotFoundError):
        resolve_prompt_path("config/prompts/planning/turn.jinja2")


def test_agent_assets_do_not_fall_back_to_the_base_package(tmp_path, monkeypatch):
    from mobilerun_agents.config.paths import resolve_prompt_path

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_prompt_path("assets/skills/README.md")


def test_local_prompt_overrides_the_bundled_default(tmp_path, monkeypatch):
    from mobilerun_agents.config.paths import resolve_prompt_path

    monkeypatch.chdir(tmp_path)
    asset = "assets/prompts/planning/turn.jinja2"
    local = tmp_path / asset
    local.parent.mkdir(parents=True)
    local.write_text("Local prompt")
    assert resolve_prompt_path(asset) == local
    assert resolve_prompt_path(local) == local

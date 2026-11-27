import asyncio

import pytest
from pydantic import ValidationError

from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import PROVIDER_KINDS, MobileConfig, OldConfigKeys


def test_llm_profiles_validate_the_same_providers_the_loader_supports():
    assert {name for name, kind in PROVIDER_KINDS.items() if kind == "llm"} == MODELS.keys()


def test_clean_config_round_trip(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "agent: {max_steps: 7, planning_max_steps: 19, planning: {subgoal_steps: 4}}\n"
        "checkpoints: {enabled: true}\nmemory: {token_limit: 9000}\n"
    )
    config = ConfigLoader.load(str(path))
    assert config.agent.max_steps == 7 and config.memory.token_limit == 9000
    assert config.agent.planning_max_steps == 19
    assert config.agent.planning.subgoal_steps == 4
    assert config.checkpoints.enabled
    assert config.profiles == {} and config.setups == {} and config.default_setup is None
    assert MobileConfig.model_validate(config.model_dump(mode="json")) == config


def test_old_top_level_memory_blocks_has_a_migration_message(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text("memory: {blocks: [app_cards]}\n")
    with pytest.raises(
        OldConfigKeys, match=r"Delete memory\.blocks; list blocks under memory\.executor\.blocks"
    ):
        MobileConfig.from_yaml(path)


def test_memory_block_defaults_and_custom_order():
    default = MobileConfig().memory
    assert default.executor.blocks == ["setup", "goal", "last_step", "history", "app_cards"]
    assert default.planner.blocks == [
        "setup",
        "goal",
        "findings",
        "last_subgoal",
        "subgoal_history",
    ]
    assert default.executor.history == default.planner.history == 5

    custom = MobileConfig.model_validate(
        {"memory": {"executor": {"blocks": ["history", "goal"]}, "planner": {"history": 2}}}
    ).memory
    assert custom.executor.blocks == ["history", "goal"]
    assert custom.planner.blocks == default.planner.blocks
    assert custom.planner.history == 2


@pytest.mark.parametrize(
    ("agent", "block", "valid"),
    [("executor", "findings", "last_step"), ("planner", "app_cards", "last_subgoal")],
)
def test_memory_rejects_block_names_unknown_to_the_agent(agent, block, valid):
    with pytest.raises(ValidationError, match=rf"Unknown memory block '{block}'.*{valid}"):
        MobileConfig.model_validate({"memory": {agent: {"blocks": [block]}}})


@pytest.mark.parametrize("agent", ["executor", "planner"])
def test_memory_history_must_be_positive(agent):
    with pytest.raises(ValidationError, match="greater than 0"):
        MobileConfig.model_validate({"memory": {agent: {"history": 0}}})


def test_run_budgets():
    config = MobileConfig().agent
    assert config.max_steps == 25
    assert config.planning_max_steps == 40
    assert config.decision.timeout == 120
    assert config.workflow_timeout is None


def test_post_action_settle_defaults_and_bounds():
    config = MobileConfig().agent
    assert config.settle_delay == 0.1
    assert MobileConfig.model_validate({"agent": {"settle_delay": 0}}).agent.settle_delay == 0
    with pytest.raises(ValidationError):
        MobileConfig.model_validate({"agent": {"settle_delay": -1}})


@pytest.mark.parametrize("value", [None, "", "override"])
def test_agent_output_directories_resolve_at_run_time(monkeypatch, tmp_path, value):
    from mobilerun_agents.config.settings import get_agent_settings

    monkeypatch.setenv("MOBILERUN_USER_CONFIG_DIR", str(tmp_path / "base"))
    for env, name in [
        ("MOBILERUN_AGENT_RUNS_DIR", "runs"),
        ("MOBILERUN_AGENT_IMAGES_DIR", "images"),
    ]:
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, str(tmp_path / name) if value else "")
        assert (
            getattr(get_agent_settings(), f"{name}_dir")
            == (tmp_path if value else tmp_path / "base") / name
        )


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


def test_cwd_prompt_does_not_override_bundled_default(tmp_path, monkeypatch):
    from mobilerun_agents.config.paths import resolve_prompt_path

    monkeypatch.chdir(tmp_path)
    asset = "assets/prompts/planning/turn.jinja2"
    local = tmp_path / asset
    local.parent.mkdir(parents=True)
    local.write_text("Local prompt")
    assert resolve_prompt_path(asset) != local
    assert resolve_prompt_path(local) == local


def test_missing_relative_prompt_does_not_resolve_from_cwd(tmp_path, monkeypatch):
    from mobilerun_agents.config.paths import resolve_prompt_path

    monkeypatch.chdir(tmp_path)
    local = tmp_path / "custom.jinja2"
    local.write_text("Local prompt")
    with pytest.raises(FileNotFoundError):
        resolve_prompt_path("custom.jinja2")


def test_render_prompt_caches_compiled_template(tmp_path, monkeypatch):
    from pathlib import Path

    from mobilerun_agents.config.prompts import render_prompt

    prompt = tmp_path / "custom.jinja2"
    prompt.write_text("Hello {{ name }}")
    read_text = Path.read_text
    reads = 0

    def counted_read_text(path, *args, **kwargs):
        nonlocal reads
        if path == prompt:
            reads += 1
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    assert render_prompt(str(prompt), name="one") == "Hello one"
    assert render_prompt(str(prompt), name="two") == "Hello two"
    assert reads == 1

"""Migration and resolution of bundled runtime assets."""

from copy import deepcopy

from mobilerun.config.migrations import migrate


def test_v8_migrates_only_bundled_paths_and_is_idempotent():
    config = {
        "_version": 7,
        "agent": {
            "fast_agent": {
                "system_prompt": "config/prompts/fast_agent/system.jinja2",
                "user_prompt": "config/prompts/custom.jinja2",
            },
            "app_cards": {"app_cards_dir": "config/app_cards"},
        },
    }
    result = migrate(config)
    assert (
        result["agent"]["fast_agent"]["system_prompt"] == "assets/prompts/fast_agent/system.jinja2"
    )
    assert result["agent"]["fast_agent"]["user_prompt"] == "config/prompts/custom.jinja2"
    assert result["agent"]["app_cards"]["app_cards_dir"] == "assets/app_cards"
    assert migrate(deepcopy(result)) == result


def test_v6_chain_keeps_manager_auto_selection():
    result = migrate(
        {
            "_version": 6,
            "agent": {"manager": {"system_prompt": "config/prompts/manager/system.jinja2"}},
        }
    )
    assert result["agent"]["manager"]["system_prompt"] is None
    assert result["_version"] == 8


def test_migration_preserves_cwd_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "config/app_cards"
    local.mkdir(parents=True)
    config = {
        "_version": 7,
        "agent": {"app_cards": {"app_cards_dir": "config/app_cards"}},
    }
    assert migrate(config)["agent"]["app_cards"]["app_cards_dir"] == "config/app_cards"


def test_legacy_app_cards_provider_falls_back_to_bundled(tmp_path, monkeypatch):
    from mobilerun.agents.context.providers.local_provider import (
        _resolve_cards_folder,
    )

    monkeypatch.chdir(tmp_path)
    assert str(_resolve_cards_folder("config/app_cards")).endswith(
        "mobilerun/agents/assets/app_cards"
    )

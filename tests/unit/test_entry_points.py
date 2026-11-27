"""The add-on advertises importable plugins for the base package."""

import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path

import click


def test_declared_entry_points_are_importable():
    config = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    groups = config["project"]["entry-points"]
    assert set(groups["mobilerun.cli"]) == {"agent", "macro-agent"}
    assert set(groups["mobilerun.operators"]) == {"agent_macro"}
    for group, entries in groups.items():
        for name, target in entries.items():
            plugin = EntryPoint(name=name, value=target, group=group).load()
            assert (
                isinstance(plugin, click.Command) if group == "mobilerun.cli" else callable(plugin)
            )


def test_agent_macro_entry_point_accepts_resolved_target(monkeypatch, tmp_path):
    import importlib

    from mobilerun_agents.config.user_config import MobileConfig

    replay_module = importlib.import_module("mobilerun_agents.cli.macro_agent.replay")

    config = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    value = config["project"]["entry-points"]["mobilerun.operators"]["agent_macro"]
    handoff = EntryPoint(name="agent_macro", value=value, group="mobilerun.operators").load()
    loaded_config = MobileConfig()
    monkeypatch.setattr(replay_module, "load_config", lambda: loaded_config)
    calls = []
    monkeypatch.setattr(replay_module, "run_replay", lambda *a, **kw: calls.append((a, kw)))
    target = object()
    options = dict(
        path=str(tmp_path / "inbox" / "v3.jsonl"),
        target=target,
        delay=0.1,
        start_from=1,
        max_steps=None,
        debug=False,
        dry_run=False,
        state_timeout=1.0,
        state_threshold=0.5,
        config_path="base-credentials.yaml",
    )
    with click.Context(click.Command("replay")) as ctx:
        ctx.meta["device_config"] = object()
        handoff(provider="openai", model="example", **options)
    assert len(calls) == 1
    config = calls[0][1].pop("config")
    assert calls[0][1] == options and not loaded_config.profiles
    assert config.profiles["recovery"].model == "example"


def test_agent_group_has_no_macro_alias():
    from mobilerun_agents.cli.agent.group import agent

    with click.Context(agent) as ctx:
        assert agent.list_commands(ctx) == ["configure", "profiles", "providers", "run", "task"]
        assert agent.get_command(ctx, "replay") is None

import json
from importlib import import_module
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from mobilerun.cli.devices.base.options import DeviceTarget

from mobilerun_agents.cli.macro_agent.group import macro_agent
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.macro_agent import MacroAgentOperator


@pytest.mark.parametrize("folder", [True, False])
def test_replay_dispatches_path_target_and_model(tmp_path, monkeypatch, folder):
    module = import_module("mobilerun_agents.cli.macro_agent.replay")
    run = Mock()
    monkeypatch.setattr(module, "run_replay", run)
    path = tmp_path if folder else tmp_path / "macro.json"
    if not folder:
        path.touch()
    result = CliRunner().invoke(
        macro_agent,
        ["replay", str(path), "emulator-5554", "--provider", "openai", "--model", "mock"],
    )
    assert result.exit_code == 0, result.exception
    assert run.call_args.args == (MacroAgentOperator,)
    options = run.call_args.kwargs
    assert options["path"] == str(path)
    assert options["target"].identifier == "emulator-5554"
    config = options["config"]
    profile = config.profiles["recovery"]
    assert (profile.provider, profile.model) == ("openai", "mock")


def test_missing_model_rejected_before_replay(tmp_path):
    result = CliRunner().invoke(
        macro_agent, ["replay", str(tmp_path), "--dry-run", "--provider", "openai"]
    )
    assert result.exit_code == 2 and "both --provider and --model" in result.output


def test_dry_run_resolves_bare_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "saved"))
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "saved" / "run-example"
    folder.mkdir(parents=True)
    (folder / "macro.json").write_text(
        json.dumps(
            {"macro_schema_version": "2.0", "actions": [{"action_type": "wait", "duration": 0}]}
        )
    )
    result = CliRunner().invoke(macro_agent, ["replay", "run-example", "--dry-run"])
    assert result.exit_code == 0, result.exception
    assert result.output == "1: wait duration=0\n"


@pytest.mark.parametrize("dry_run", [True, False])
def test_invalid_agent_config_only_blocks_live_replay(tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(DeviceTarget, "open", lambda target: pytest.fail("device opened"))
    config = ConfigLoader.get_user_config_path()
    config.parent.mkdir(parents=True)
    config.write_text("profiles: [", encoding="utf-8")
    macro = tmp_path / "macro.json"
    macro.write_text(
        json.dumps(
            {"macro_schema_version": "2.0", "actions": [{"action_type": "wait", "duration": 0}]}
        ),
        encoding="utf-8",
    )

    args = [str(macro), "--dry-run"] if dry_run else [str(macro), "emulator-5554"]
    result = CliRunner().invoke(macro_agent, ["replay", *args])

    if dry_run:
        assert result.exit_code == 0, result.exception
        assert result.output == "1: wait duration=0\n"
    else:
        assert result.exit_code == 1
        assert result.output == "Error: Invalid agents.yaml syntax\n"

from importlib import import_module
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from mobilerun.cli.devices.base.options import DeviceTarget

from mobilerun_agents.cli.macro_agent.group import macro_agent
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.macro_agent import MacroAgentOperator


@pytest.mark.parametrize("folder", [True, False])
def test_replay_dispatches_path_target_and_model(write_macro, monkeypatch, folder):
    module = import_module("mobilerun_agents.cli.macro_agent.replay")
    run = Mock()
    monkeypatch.setattr(module, "run_replay", run)
    pinned = write_macro([{"i": 1, "action": "wait", "args": {"duration": 0}}])
    path = pinned.parent if folder else pinned
    result = CliRunner().invoke(
        macro_agent,
        ["replay", str(path), "emulator-5554", "--provider", "openai", "--model", "mock"],
    )
    assert result.exit_code == 0, result.exception
    assert run.call_args.args == (MacroAgentOperator,)
    options = run.call_args.kwargs
    assert options["path"] == str(pinned)
    assert options["target"].identifier == "emulator-5554"
    config = options["config"]
    profile = config.profiles["recovery"]
    assert (profile.provider, profile.model) == ("openai", "mock")


def test_missing_model_rejected_before_replay(write_macro):
    path = write_macro([{"i": 1, "action": "wait", "args": {"duration": 0}}])
    result = CliRunner().invoke(
        macro_agent, ["replay", str(path), "--dry-run", "--provider", "openai"]
    )
    assert result.exit_code == 2 and "both --provider and --model" in result.output


@pytest.mark.parametrize("target", ["name", "folder", "file"])
def test_dry_run_resolves_latest_numeric_or_pinned_version(write_macro, target):
    pinned = write_macro([{"i": 1, "action": "wait", "args": {"duration": 0}}])
    for version in (9, 10):
        write_macro([{"i": 1, "action": "wait", "args": {"duration": version}}], version)
    value = {"name": "inbox", "folder": str(pinned.parent), "file": str(pinned)}[target]
    result = CliRunner().invoke(macro_agent, ["replay", value, "--dry-run"])
    assert result.exit_code == 0, result.exception
    assert result.output == f"1: wait duration={0 if target == 'file' else 10}\n"


@pytest.mark.parametrize("dry_run", [True, False])
def test_invalid_agent_config_only_blocks_live_replay(write_macro, monkeypatch, dry_run):
    monkeypatch.setattr(DeviceTarget, "open", lambda target: pytest.fail("device opened"))
    config = ConfigLoader.get_user_config_path()
    config.parent.mkdir(parents=True)
    config.write_text("profiles: [", encoding="utf-8")
    macro = write_macro([{"i": 1, "action": "wait", "args": {"duration": 0}}])

    args = [str(macro), "--dry-run"] if dry_run else [str(macro), "emulator-5554"]
    result = CliRunner().invoke(macro_agent, ["replay", *args])

    if dry_run:
        assert result.exit_code == 0, result.exception
        assert result.output == "1: wait duration=0\n"
    else:
        assert result.exit_code == 1
        assert result.output == "Error: Invalid agents.yaml syntax\n"


def test_direct_agent_replay_preflights_commerce_before_opening(write_macro, monkeypatch):
    module = import_module("mobilerun.macro.cli.replay")
    path = write_macro(
        [{"i": 1, "action": "extension_call", "args": {"name": "sms.send", "arguments": {}}}]
    )
    confirmations = []

    async def reject(label, *, yes):
        confirmations.append((label, yes))
        raise RuntimeError("not authorized")

    monkeypatch.setattr(module, "confirm_paid_action", reject)
    monkeypatch.setattr(DeviceTarget, "open", lambda target: pytest.fail("device opened"))
    result = CliRunner().invoke(macro_agent, ["replay", str(path), "emulator-5554"])
    assert str(result.exception) == "not authorized"
    assert confirmations == [("Send 1 SMS", False)]

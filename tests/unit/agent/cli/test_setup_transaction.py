import pytest
from click.testing import CliRunner
from mobilerun.cli.base import menu, prompts

from mobilerun_agents.agent.auth import store
from mobilerun_agents.cli.agent import agent
from mobilerun_agents.config.loader import ConfigLoader


@pytest.mark.parametrize("action", ["new", "replace", "remove"])
def test_failed_config_save_preserves_config_and_credentials(action, monkeypatch):
    runner = CliRunner()
    args = ["setup", "--provider", "openai", "--api-key-stdin"]
    path = ConfigLoader.get_user_config_path()
    if action != "new":
        assert runner.invoke(agent, args, input="fixture-original\n").exit_code == 0
    original = path.read_bytes() if path.exists() else None

    def fail_save(config):
        raise OSError("simulated config write failure")

    monkeypatch.setattr(ConfigLoader, "save", fail_save)
    if action == "remove":
        choices = iter(["Remove provider", "openai", "Done"])
        monkeypatch.setattr(menu, "is_interactive", lambda: True)
        monkeypatch.setattr(prompts, "select_prompt", lambda *_: next(choices))
        args = ["setup"]
    result = runner.invoke(agent, args, input="fixture-replacement\n")
    assert result.exit_code != 0 and "fixture-" not in result.output
    assert (path.read_bytes() if path.exists() else None) == original
    record = store.load("openai")
    assert (record.api_key if record else None) == (None if action == "new" else "fixture-original")


def test_second_key_failure_rolls_back_first_key_and_exact_config(monkeypatch):
    from mobilerun_agents.cli.agent.setup import _save

    config = ConfigLoader.load()
    path = ConfigLoader.save(config)
    original = path.read_bytes() + b"\n# Preserve manual comments\n"
    path.write_bytes(original)
    config.agent.max_steps += 1
    store.save_key("openai", "fixture-original")

    def fail_delete(provider):
        raise OSError("simulated key write failure")

    monkeypatch.setattr(store, "delete", fail_delete)
    with pytest.raises(OSError):
        _save(config, {"openai": "fixture-changed", "anthropic": None})
    assert path.read_bytes() == original
    assert store.load("openai").api_key == "fixture-original"

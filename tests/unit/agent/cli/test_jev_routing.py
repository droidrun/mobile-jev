import asyncio

import pytest
from click.testing import CliRunner
from mobilerun.cli.base import menu, prompts

from mobilerun_agents.agent.action_selector.decision_model.base import DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.agent.action_selector.delegating import DelegatingSelector
from mobilerun_agents.agent.action_selector.llm.llama_index import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.loader import default_planner, default_selector
from mobilerun_agents.agent.auth import store
from mobilerun_agents.agent.planner import LlamaIndexPlanner
from mobilerun_agents.cli.agent import agent
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import DelegationConfig


@pytest.mark.parametrize("only", [False, True])
def test_activate_saved_jev_profile_and_route_each_role(monkeypatch, only):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    runner = CliRunner()
    assert (
        runner.invoke(agent, ["setup", "--provider", "ollama", "--model", "local"]).exit_code == 0
    )
    args = ["setup", "--provider", "typesafe"] + (["--jev-only"] if only else [])
    assert runner.invoke(agent, args).exit_code == 0
    assert ConfigLoader.load().agent.selector == "ollama"
    assert runner.invoke(agent, ["setup", "--default", "typesafe"]).exit_code == 0
    for role in ["agent", "executor"]:
        selector = asyncio.run(default_selector(role))
        if not only:
            assert type(selector) is DelegatingSelector
            assert selector.min_confidence == DelegationConfig().min_confidence
            assert type(selector.delegate) is LlamaIndexLLMSelector
            assert selector.delegate.shared_llm.model == "local"
            selector = selector.primary
        assert type(selector) is DecisionSelector and type(selector.model) is TypeSafeDecisionModel
    planner = asyncio.run(default_planner())
    assert type(planner) is LlamaIndexPlanner and planner.llm.metadata.model_name == "local"


def test_cancel_discards_jev_profile_and_key(monkeypatch):
    from importlib import import_module

    monkeypatch.setattr(menu, "is_interactive", lambda: True)
    wizard = import_module("mobilerun_agents.cli.agent.wizard")
    monkeypatch.setattr(wizard, "read_secret", lambda: "dummy-example-key")
    answers = iter(["Add provider", "typesafe", "API key", "Jev only", None])
    monkeypatch.setattr(prompts, "select_prompt", lambda *_: next(answers))
    result = CliRunner().invoke(agent, ["setup"])
    assert result.exit_code == 0 and "cancelled" in result.output
    assert "dummy-example-key" not in result.output
    assert not ConfigLoader.get_user_config_path().exists() and not store.list_providers()

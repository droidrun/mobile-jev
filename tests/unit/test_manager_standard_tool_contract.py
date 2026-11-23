import ast
from pathlib import Path

import pytest

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.workflows.manager.manager_agent import ManagerAgent
from mobilerun.agents.workflows.manager.stateless_manager_agent import StatelessManagerAgent


@pytest.mark.parametrize("manager_type", [ManagerAgent, StatelessManagerAgent])
def test_manager_declares_standard_tool_names(manager_type):
    manager = manager_type(
        llm=None, action_ctx=None, state_provider=None, agent_config=AgentConfig()
    )
    assert manager.standard_tool_names == []
    manager.standard_tool_names = ["tap"]
    assert manager.standard_tool_names == ["tap"]


def test_droid_wires_standard_tool_names_without_manager_kind_check():
    path = Path(
        __import__("mobilerun.agents.workflows.droid.droid_agent", fromlist=["__file__"]).__file__
    )
    tree = ast.parse(path.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "ManagerAgent"
        for node in ast.walk(tree)
    )

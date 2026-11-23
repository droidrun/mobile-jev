import asyncio
from types import SimpleNamespace

import pytest

from mobilerun.agents.config.user_config import AgentConfig
from mobilerun.agents.workflows.executor.executor_agent import ExecutorAgent
from mobilerun.agents.workflows.manager.manager_agent import ManagerAgent
from mobilerun.agents.workflows.manager.stateless_manager_agent import StatelessManagerAgent


@pytest.mark.parametrize("manager_cls", [ManagerAgent, StatelessManagerAgent])
def test_manager_requires_config(manager_cls):
    with pytest.raises(ValueError, match="agent configuration"):
        manager_cls(llm=SimpleNamespace(), action_ctx=None, state_provider=None)


@pytest.mark.parametrize("manager_cls", [ManagerAgent, StatelessManagerAgent])
def test_manager_context_requires_device_dependencies(manager_cls):
    manager = manager_cls(
        llm=SimpleNamespace(),
        action_ctx=None,
        state_provider=None,
        agent_config=AgentConfig(),
    )
    with pytest.raises(RuntimeError, match="action context and state provider"):
        asyncio.run(manager.prepare_context(SimpleNamespace(), SimpleNamespace()))


def test_executor_context_requires_registry():
    executor = ExecutorAgent(
        llm=SimpleNamespace(),
        registry=None,
        action_ctx=None,
        shared_state=SimpleNamespace(),
        agent_config=AgentConfig(),
    )
    with pytest.raises(RuntimeError, match="tool registry"):
        asyncio.run(executor.prepare_context(SimpleNamespace(), SimpleNamespace()))

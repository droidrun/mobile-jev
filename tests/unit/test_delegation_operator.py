"""Fake-only contracts for the operator entry points."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.operators.agent import AgentOperator
from mobilerun.agents.operators.delegation import DelegationOperator


def test_delegation_forces_reasoning_without_mutating_config(monkeypatch):
    config = MobileConfig()
    operator = DelegationOperator(config=config)
    captured = []

    async def capture(self, *args, **kwargs):
        captured.append(self.config)

    monkeypatch.setattr(AgentOperator, "execute", capture)
    asyncio.run(operator.execute(object(), "task"))
    assert captured[0].agent.reasoning is True
    assert captured[0] is not config
    assert config.agent.reasoning is False


def test_delegation_loads_config_at_execute_time(monkeypatch):
    loader = Mock(side_effect=lambda: MobileConfig())
    provider = Mock()
    monkeypatch.setattr("mobilerun.agents.operators.delegation.ConfigLoader.load", loader)
    monkeypatch.setattr("mobilerun.agents.operators.agent.load_agent_llms", provider)
    monkeypatch.setattr(AgentOperator, "execute", AsyncMock())
    operator = DelegationOperator()
    loader.assert_not_called()
    provider.assert_not_called()
    asyncio.run(operator.execute(object(), "first"))
    asyncio.run(operator.execute(object(), "second"))
    assert loader.call_count == 2
    provider.assert_not_called()


def test_delegation_forwards_options_and_result(monkeypatch):
    config, llms, context, device, result = MobileConfig(), object(), object(), object(), object()
    calls = []

    class Output(BaseModel):
        value: str

    async def capture(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return result

    monkeypatch.setattr(AgentOperator, "execute", capture)
    operator = DelegationOperator(config=config, llms=llms, context_provider=context)
    actual = asyncio.run(operator.execute(device, "task", output=Output, max_steps=4, timeout=5.0))
    assert actual is result
    delegated, args, kwargs = calls[0]
    assert delegated.llms is llms and delegated.context_provider is context
    assert args == (device, "task")
    assert kwargs == {"output": Output, "max_steps": 4, "timeout": 5.0}


def test_delegation_forwards_failure_without_mutation(monkeypatch):
    config = MobileConfig()
    error = RuntimeError("sentinel")
    monkeypatch.setattr(AgentOperator, "execute", AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(DelegationOperator(config=config).execute(object(), "task"))
    assert caught.value is error
    assert config.agent.reasoning is False


@pytest.mark.parametrize("reasoning", [False, True])
def test_agent_operator_keeps_both_modes(monkeypatch, reasoning):
    import mobilerun.agents.operators.agent as agent_module

    config = MobileConfig()
    config.agent.reasoning = reasoning
    captured = []

    class FakeAgent:
        trajectory = None

        def __init__(self, **kwargs):
            captured.append(kwargs["config"])

        async def run(self):
            return SimpleNamespace(structured_output=None, answer="ok", success=True, steps=1)

    monkeypatch.setattr(agent_module, "MobileAgent", FakeAgent)
    monkeypatch.setattr(agent_module, "ensure_agent_logging", lambda: None)
    monkeypatch.setattr(agent_module, "apply_log_level", lambda _: None)
    asyncio.run(AgentOperator(config=config, llms=object()).execute(object(), "task"))
    assert captured[0].agent.reasoning is reasoning

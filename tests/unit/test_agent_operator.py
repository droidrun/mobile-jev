"""Operator calls isolate overrides and expose serializable result paths."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.operators import agent as module


@pytest.fixture
def calls(monkeypatch):
    configs = []

    class FakeAgent:
        def __init__(self, *, config, **kwargs):
            self.config = config
            self.trajectory = SimpleNamespace(trajectory_folder=Path("trajectories/run"))
            configs.append(config)

        async def run(self):
            # Yield so concurrent invocations overlap before observing the value.
            await asyncio.sleep(0)
            return SimpleNamespace(
                success=True,
                reason=str(self.config.agent.max_steps),
                answer=None,
                structured_output=None,
                steps=3,
            )

    monkeypatch.setattr(module, "MobileAgent", FakeAgent)
    return configs


def test_sequential_overrides_do_not_leak_into_operator_or_caller(calls):
    config = MobileConfig()
    operator = module.AgentOperator(config=config, llms={})

    async def run():
        first = await operator.execute(object(), "first", max_steps=2)
        second = await operator.execute(object(), "second")
        assert (first.answer, second.answer) == ("2", "15")

    asyncio.run(run())
    assert config.agent.max_steps == 15
    assert calls[0] is not config and calls[1] is not config
    assert calls[0].agent is not calls[1].agent


def test_concurrent_calls_have_independent_overrides(calls):
    config = MobileConfig()
    operator = module.AgentOperator(config=config, llms={})

    async def run():
        results = await asyncio.gather(
            operator.execute(object(), "first", max_steps=2),
            operator.execute(object(), "second", max_steps=7),
            operator.execute(object(), "third"),
        )
        assert [result.answer for result in results] == ["2", "7", "15"]

    asyncio.run(run())
    assert config.agent.max_steps == 15


def test_trajectory_path_is_a_string(calls):
    result = asyncio.run(
        module.AgentOperator(config=MobileConfig(), llms={}).execute(object(), "go")
    )
    assert result.success
    assert result.trajectory_path == "trajectories/run"
    assert result.steps == 3


def test_result_contract_requires_steps(monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.trajectory = None

        async def run(self):
            return SimpleNamespace(success=True, reason="done", answer=None, structured_output=None)

    monkeypatch.setattr(module, "MobileAgent", FakeAgent)
    with pytest.raises(AttributeError, match="steps"):
        asyncio.run(module.AgentOperator(config=MobileConfig(), llms={}).execute(object(), "go"))


def test_base_configuration_is_parsed_at_the_agent_boundary(calls):
    from mobilerun.config.user_config import MobileConfig as BaseMobileConfig

    config = BaseMobileConfig.from_dict({"agent": {"max_steps": 9}})
    agent_config = MobileConfig.from_base(config)
    result = asyncio.run(
        module.AgentOperator(config=agent_config, llms={}).execute(object(), "go", max_steps=2)
    )
    assert result.answer == "2"
    assert calls[0].agent.max_steps == 2
    assert config.agent == {"max_steps": 9}
    result = asyncio.run(module.AgentOperator(config=agent_config, llms={}).execute(object(), "go"))
    assert result.answer == "9"
    assert config.agent == {"max_steps": 9}


def test_agent_configuration_subclass_keeps_custom_state(calls):
    class CustomConfig(MobileConfig):
        extra_setting = "custom"

    config = CustomConfig()
    config.agent.max_steps = 8
    result = asyncio.run(module.AgentOperator(config=config, llms={}).execute(object(), "go"))

    assert result.answer == "8"
    assert type(calls[0]) is CustomConfig
    assert calls[0].extra_setting == "custom"
    assert calls[0] is not config


def test_shared_first_run_defaults_preserve_agent_streaming():
    from mobilerun.config.user_config import MobileConfig as BaseMobileConfig

    assert MobileConfig.from_dict(BaseMobileConfig().to_dict()).agent.streaming is True


def test_operator_returns_result_answer_before_reason(monkeypatch):
    class FakeAgent:
        trajectory = None

        def __init__(self, **kwargs):
            pass

        async def run(self):
            return SimpleNamespace(
                success=True,
                reason="Completed",
                answer="The current balance is $42.",
                structured_output=None,
                steps=1,
            )

    monkeypatch.setattr(module, "MobileAgent", FakeAgent)
    result = asyncio.run(
        module.AgentOperator(config=MobileConfig(), llms={}).execute(object(), "balance")
    )
    assert result.answer == "The current balance is $42."


def test_external_operator_does_not_load_builtin_llms(monkeypatch, calls):
    config = MobileConfig()
    config.agent.name = "external"

    def fail(**kwargs):
        raise AssertionError("built-in LLMs must not be loaded")

    monkeypatch.setattr(module, "load_agent_llms", fail)
    result = asyncio.run(module.AgentOperator(config=config).execute(object(), "go"))
    assert result.success

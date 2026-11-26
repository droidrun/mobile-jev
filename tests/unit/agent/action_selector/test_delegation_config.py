import asyncio

import pytest
from llama_index.core.llms import MockLLM

from mobilerun_agents.agent.action_selector import (
    DecisionSelector,
    DelegatingSelector,
    LlamaIndexLLMSelector,
    loader,
)
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import (
    DecisionProfile,
    DelegationConfig,
    LLMProfile,
    MobileConfig,
)


@pytest.mark.parametrize("primary", [None, "decision"])
def test_typesafe_primary_builds_only_delegate_llm(primary, monkeypatch):
    config = MobileConfig()
    config.agent.selector = "selected"
    config.decision_profiles[primary or config.agent.selector] = DecisionProfile()
    config.llm_profiles["helper"] = LLMProfile(provider="openai", model="helper-model")
    config.agent.delegation = DelegationConfig(
        primary=primary, delegate="helper", min_confidence=0.8
    )
    shared = MockLLM()
    calls = []

    async def fake_build(provider, model, **options):
        calls.append((provider, model))
        return shared

    monkeypatch.setattr(loader, "build_llm", fake_build)
    selector = asyncio.run(loader.build_selector(config))
    assert calls == [("openai", "helper-model")]
    assert type(selector) is DelegatingSelector
    assert type(selector.primary) is DecisionSelector
    assert selector.shared_llm is shared
    assert selector.min_confidence == 0.8


@pytest.mark.parametrize("role", ["agent", "executor"])
@pytest.mark.parametrize("vision", [None, "only"])
def test_default_jev_and_llm_use_profile_vision_or_run_override(monkeypatch, role, vision):
    config = MobileConfig(
        default_llm="local",
        llm_profiles={"local": LLMProfile(provider="ollama", model="fake")},
        decision_profiles={"jev": DecisionProfile()},
    )
    config.agent.selector = "jev"
    config.agent.delegation = DelegationConfig(delegate="local")
    path = ConfigLoader.save(config)
    saved = path.read_bytes()

    async def fake_build(*args, **kwargs):
        return MockLLM()

    monkeypatch.setattr(loader, "build_llm", fake_build)
    selector = asyncio.run(loader.default_selector(role, vision))
    assert type(selector) is DelegatingSelector
    assert type(selector.primary) is DecisionSelector
    assert type(selector.primary.model) is TypeSafeDecisionModel
    assert type(selector.delegate) is LlamaIndexLLMSelector
    assert selector.min_confidence == 0.7
    assert (selector.primary.vision, selector.delegate.vision) == (vision or "on", vision or "off")
    assert path.read_bytes() == saved and b"vision:" not in saved
    planner = asyncio.run(loader.default_planner())
    assert type(planner) is loader.LlamaIndexPlanner and type(planner.llm) is MockLLM


@pytest.mark.parametrize("provider,expected", [("openai", "on"), ("ollama", "off")])
def test_unset_profile_vision_resolves_to_provider_default_offline(monkeypatch, provider, expected):
    async def fake_build(*args, **kwargs):
        return MockLLM()

    monkeypatch.setattr(loader, "build_llm", fake_build)
    config = MobileConfig(llm_profiles={"default": LLMProfile(provider=provider, model="unlisted")})
    selector = asyncio.run(loader.build_selector(config))
    assert type(selector) is LlamaIndexLLMSelector
    assert selector.vision == expected

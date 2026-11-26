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
    LLMProfile,
    MobileConfig,
    SetupConfig,
)


def test_typesafe_primary_builds_only_delegate_llm(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    config = MobileConfig(
        profiles={
            "selected": DecisionProfile(provider="typesafe", min_confidence=0.8),
            "helper": LLMProfile(provider="openai", model="helper-model"),
        },
        setups={"jev": SetupConfig(selector="selected", delegate="helper")},
    )
    shared = MockLLM()
    calls = []

    async def fake_build(provider, model, **options):
        calls.append((provider, model))
        return shared

    monkeypatch.setattr(loader, "build_llm", fake_build)
    with ConfigLoader.override(config):
        selector = asyncio.run(loader.build_setup("jev")).selector
    assert calls == [("openai", "helper-model")]
    assert type(selector) is DelegatingSelector
    assert type(selector.primary) is DecisionSelector
    assert selector.shared_llm is shared
    assert selector.min_confidence == 0.8


@pytest.mark.parametrize("role", ["agent", "executor"])
@pytest.mark.parametrize("vision", [None, "only"])
def test_default_jev_and_llm_use_profile_vision_or_run_override(monkeypatch, role, vision):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake")
    config = MobileConfig(
        profiles={
            "local": LLMProfile(provider="ollama", model="fake"),
            "jev": DecisionProfile(provider="typesafe"),
        },
        setups={
            "straight": SetupConfig(selector="jev", delegate="local"),
            "planned": SetupConfig(planner="local", executor="jev", delegate="local"),
        },
        default_setup="straight" if role == "agent" else "planned",
    )
    path = ConfigLoader.save(config)
    saved = path.read_bytes()

    async def fake_build(*args, **kwargs):
        return MockLLM()

    monkeypatch.setattr(loader, "build_llm", fake_build)
    spec = asyncio.run(loader.build_setup(None, vision))
    selector = spec.selector if role == "agent" else spec.executor
    assert type(selector) is DelegatingSelector
    assert type(selector.primary) is DecisionSelector
    assert type(selector.primary.model) is TypeSafeDecisionModel
    assert type(selector.delegate) is LlamaIndexLLMSelector
    assert selector.min_confidence == 0.7
    assert (selector.primary.vision, selector.delegate.vision) == (vision or "on", vision or "off")
    assert path.read_bytes() == saved and b"vision:" not in saved
    if role == "executor":
        assert type(spec.planner) is loader.LlamaIndexPlanner and type(spec.planner.llm) is MockLLM


@pytest.mark.parametrize("provider,expected", [("openai", "on"), ("ollama", "off")])
def test_unset_profile_vision_resolves_to_provider_default_offline(monkeypatch, provider, expected):
    async def fake_build(*args, **kwargs):
        return MockLLM()

    monkeypatch.setattr(loader, "build_llm", fake_build)
    config = MobileConfig(profiles={"default": LLMProfile(provider=provider, model="unlisted")})
    with ConfigLoader.override(config):
        selector = asyncio.run(loader.default_setup()).selector
    assert type(selector) is LlamaIndexLLMSelector
    assert selector.vision == expected

import pytest
from pydantic import ValidationError

from mobilerun_agents.agent.action_selector.decision_model.loader import load_decision_model
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.config.user_config import DecisionConfig, DecisionProfile, MobileConfig


def test_decision_profile_builds_without_credentials_using_run_config():
    profile = DecisionProfile(model="jev", kwargs={"config": DecisionConfig(max_options=2)})
    model = load_decision_model(profile, DecisionConfig(max_options=12))
    assert type(model) is TypeSafeDecisionModel
    assert model.max_options == 12


def test_decision_config_roundtrip():
    data = {
        "decision_profiles": {"fast_agent": {"provider": "typesafe", "model": "jev"}},
        "agent": {"delegation": {"delegate": "helper", "min_confidence": 0.8}},
    }
    config = MobileConfig.model_validate(data)
    assert config.decision_profiles["fast_agent"].model == "jev"
    assert config.agent.delegation.delegate == "helper"
    assert (
        MobileConfig.model_validate(config.model_dump(mode="json")).agent.delegation.min_confidence
        == 0.8
    )


def test_untrusted_delegation_config_validates():
    with pytest.raises(ValidationError):
        MobileConfig.model_validate({"agent": {"delegation": {"min_confidence": 4}}})

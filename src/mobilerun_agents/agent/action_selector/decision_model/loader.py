"""Build decision models. Each model resolves credentials at request time."""

from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.config.user_config import DecisionConfig, DecisionProfile


def load_decision_model(profile: DecisionProfile, config: DecisionConfig) -> TypeSafeDecisionModel:
    options = dict(profile.kwargs)
    if profile.base_url:
        options["base_url"] = profile.base_url
    if profile.model is not None:
        options["model"] = profile.model
    options["config"] = config
    return TypeSafeDecisionModel(**options)

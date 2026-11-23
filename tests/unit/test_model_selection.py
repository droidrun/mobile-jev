from mobilerun.agents.config.user_config import LLMProfile, MobileConfig
from mobilerun.agents.llm.providers import model_selection


def test_profile_settings_and_vision_mutate_config():
    config = MobileConfig()
    profile = config.llm_profiles["manager"]
    old_kwargs = profile.kwargs
    model_selection.set_profile_max_tokens(profile, 512)
    model_selection.set_profile_context_window(profile, 4096)
    assert profile.kwargs == {"max_tokens": 512, "context_window": 4096}
    assert old_kwargs == {}
    model_selection.set_vision_all(config, False)
    assert not model_selection.is_vision_enabled(config)
    model_selection.set_vision_all(config, True)
    assert all(
        (
            config.agent.manager.vision,
            config.agent.executor.vision,
            config.agent.fast_agent.vision,
        )
    )
    config.llm_profiles["manager"] = LLMProfile(provider="Ollama", provider_family="minimax")
    assert model_selection.any_ollama_profile(config)
    assert model_selection.any_minimax_profile(config)


def test_apply_model_selection_updates_all_roles():
    from mobilerun.agents.llm.providers.registry import resolve_provider_variant

    config = MobileConfig()
    variant = resolve_provider_variant("openai", "oauth")
    model_selection.apply_model_selection(
        config,
        family_id="openai",
        variant=variant,
        selected_auth_mode="oauth",
        selected_model="gpt-5.4",
        selected_api_key=None,
        selected_api_key_source=None,
        selected_base_url=None,
        credential_path="/example/auth.json",
    )
    for profile in config.llm_profiles.values():
        assert profile.model == "gpt-5.4"
        assert profile.provider_family == "openai"
        assert profile.credential_path == "/example/auth.json"

import warnings

from mobilerun.config.migrations import CURRENT_VERSION

from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.llm.llm_picker import load_llm


def test_explicit_saved_gemini_3_1_flash_lite_profile_remains_usable_and_silent():
    config = MobileConfig.from_dict(
        {
            "_version": CURRENT_VERSION,
            "llm_profiles": {
                "manager": {
                    "provider": "GoogleGenAI",
                    "model": "gemini-3.1-flash-lite",
                    "temperature": 0.2,
                    "kwargs": {
                        "api_key": "test-key",
                        "context_window": 1_000_000,
                        "max_tokens": 1_024,
                    },
                }
            },
        }
    )
    profile = config.llm_profiles["manager"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        llm = load_llm(profile.provider, **profile.to_load_llm_kwargs())

    assert profile.model == "gemini-3.1-flash-lite"
    assert llm.model == "gemini-3.1-flash-lite"
    assert not any("gemini-3.1-flash-lite" in str(warning.message) for warning in caught)

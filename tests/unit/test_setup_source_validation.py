import pytest

from mobilerun.agents.llm.providers.registry import resolve_provider_variant
from mobilerun.agents.llm.providers.setup_service import (
    SetupSelection,
    create_profile_for_variant,
)


def test_invalid_api_key_source_is_rejected():
    variant = resolve_provider_variant("minimax", "api_key")
    selection = SetupSelection(
        family_id="minimax",
        variant_id=variant.id,
        auth_mode="api_key",
        model=variant.models[0],
        api_key_source="invalid",
    )
    with pytest.raises(ValueError, match="Invalid API key source"):
        create_profile_for_variant(variant, selection)

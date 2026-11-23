import click
import pytest

from mobilerun.agents.cli.agent.llm import wizard
from mobilerun.agents.llm.providers.registry import resolve_provider_variant


def test_escape_cancels_provider_selection(monkeypatch):
    monkeypatch.setattr(wizard, "select_prompt", lambda *args, **kwargs: None)
    with pytest.raises(click.Abort):
        wizard._select_with_back("Choose", [])


def test_escape_cancels_minimax_region_selection(monkeypatch):
    monkeypatch.setattr(wizard, "select_prompt", lambda *args, **kwargs: None)
    variant = resolve_provider_variant("minimax", "api_key")
    with pytest.raises(click.Abort):
        wizard._prompt_base_url_for_variant(variant)


def test_back_from_api_key_source_returns_back_marker(monkeypatch):
    variant = resolve_provider_variant("minimax", "api_key")
    monkeypatch.setattr(wizard, "_prompt_api_key_source", lambda variant: wizard._BACK)
    key, source = wizard._prompt_api_key_for_variant(variant)
    assert key == wizard._BACK
    assert source == wizard._BACK

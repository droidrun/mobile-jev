import asyncio
import json
import logging

import httpx
import pytest

from mobilerun.agents.context.providers.local_provider import LocalCardBackend
from mobilerun.agents.context.providers.server_provider import ServerCardBackend


def test_local_transient_read_failure_is_not_cached(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    folder = tmp_path / "android"
    folder.mkdir()
    (folder / "app_cards.json").write_text(json.dumps({"app": "card.md"}))
    card = folder / "card.md"
    card.write_text("context")
    backend = LocalCardBackend(str(tmp_path))
    original = type(card).read_text
    calls = 0

    def read(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(card), "read_text", read)
    assert asyncio.run(backend.fetch_card("app", platform="android")) == ""
    assert asyncio.run(backend.fetch_card("app", platform="android")) == "context"
    assert "Failed to load app card" in caplog.text


def test_server_failure_is_not_cached(monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    calls = 0

    async def post(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"app_card": "context"})

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    backend = ServerCardBackend("https://example.invalid", max_retries=1)
    assert asyncio.run(backend.fetch_card("app")) == ""
    assert asyncio.run(backend.fetch_card("app")) == "context"
    assert "Server request failed" in caplog.text


def test_server_programmer_errors_propagate(monkeypatch):
    async def post(self, *args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    backend = ServerCardBackend("https://example.invalid", max_retries=1)
    with pytest.raises(TypeError, match="bug"):
        asyncio.run(backend.fetch_card("app"))


def test_local_mapping_failure_is_retried(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    folder = tmp_path / "android"
    folder.mkdir()
    mapping = folder / "app_cards.json"
    mapping.write_text("{")
    (folder / "card.md").write_text("context")
    backend = LocalCardBackend(str(tmp_path))
    assert asyncio.run(backend.fetch_card("app", platform="android")) == ""
    mapping.write_text(json.dumps({"app": "card.md"}))
    assert asyncio.run(backend.fetch_card("app", platform="android")) == "context"
    assert "Failed to load" in caplog.text

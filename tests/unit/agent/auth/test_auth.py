import socket
from dataclasses import fields

import pytest
from mobilerun.errors import CredentialFormatError

from mobilerun_agents.agent.auth import Credential, CredentialMissing, resolve, store


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("auth unit tests cannot open network connections")

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, blocked)


@pytest.mark.anyio
async def test_priority_and_missing(monkeypatch):
    with pytest.raises(CredentialMissing) as missing:
        await resolve("typesafe")
    assert missing.value.args[0] == (
        "No API key for typesafe: pass api_key or set TYPESAFE_API_KEY. "
        "You can also save a provider key with `mobilerun agent setup`."
    )
    store.save_key("typesafe", "saved")
    assert await resolve("typesafe") == Credential("typesafe", api_key="saved")
    monkeypatch.setenv("TYPESAFE_API_KEY", "env")
    assert (await resolve("typesafe")).api_key == "env"
    assert (await resolve("typesafe", "explicit")).api_key == "explicit"


@pytest.mark.anyio
async def test_only_api_keys_are_accepted(tmp_path):
    assert [field.name for field in fields(Credential)] == ["provider", "api_key"]
    store.save_key("openai", "private")
    assert "private" not in repr(await resolve("openai"))
    (tmp_path / "config/agent/auth/openai.json").write_text('{"access_token":"private"}')
    with pytest.raises(CredentialFormatError) as error:
        await resolve("openai")
    assert "private" not in str(error.value)


def test_store_atomic_private(tmp_path, monkeypatch):
    store.save_key("openai", "before")
    (path,) = (tmp_path / "config/agent/auth").iterdir()
    assert path.stat().st_mode & 0o777 == 0o600

    def fail_replace(src, dst):
        assert src.stat().st_mode & 0o777 == 0o600
        raise OSError("disk failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.save_key("openai", "after")
    assert store.load("openai").api_key == "before"
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(ValueError):
        store.save_key("../escape", "after")


def test_enumerate_and_delete_saved_providers(tmp_path):
    assert store.list_providers() == []
    for name in ("openai", "Custom"):
        store.save_key(name, "private")
    folder = tmp_path / "config/agent/auth"
    (folder / "invalid-name.json").touch()
    (folder / "directory.json").mkdir()
    assert store.list_providers() == ["custom", "openai"]
    store.delete("CUSTOM")
    store.delete("custom")
    assert not (folder / "custom.json").exists()
    assert store.list_providers() == ["openai"]

import json
import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Macro unit tests must not use network connections")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)


@pytest.fixture
def macros(tmp_path, monkeypatch):
    root = tmp_path / "macros"
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(root))
    monkeypatch.setenv("MOBILERUN_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_RUN_SETTLE_SECONDS", "0")
    return root


@pytest.fixture
def write_macro(macros):
    def write(rows, version=3):
        path = macros / "inbox" / f"v{version}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    return write

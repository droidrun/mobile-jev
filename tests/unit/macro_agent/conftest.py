import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Macro unit tests must not use network connections")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "getaddrinfo", reject)

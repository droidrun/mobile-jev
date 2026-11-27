import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("selector unit tests must not use the network")

    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, method, deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

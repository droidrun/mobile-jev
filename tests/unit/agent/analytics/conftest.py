import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def deny(*args, **kwargs):
        pytest.fail("analytics unit tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)

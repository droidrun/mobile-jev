"""Credentials and Android SDK are restored only for explicitly enabled smoke."""

import pytest


@pytest.fixture
def android_sdk_for_smoke(monkeypatch, real_android_sdk, isolate_user_configuration):
    monkeypatch.setenv("ANDROID_HOME", real_android_sdk)

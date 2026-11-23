import asyncio

import pytest
from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)
from mobilerun_core_local.driver.android import AndroidPortalHttpDriver

from mobilerun.agents.config.user_config import (
    DeviceConfig,
    MobileConfig,
    TelemetryConfig,
)

PORTAL_URL = "http://192.168.1.20:8080"


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("http://192.168.1.20:8080", True),
        ("HTTPS://phone.local:8080", True),
        ("emulator-5554", False),
        ("192.168.1.20:5555", False),
        ("", False),
        (None, False),
    ],
)
def test_is_portal_http_url(device, expected):
    assert is_portal_http_url(device) is expected


def test_portal_http_driver_requires_token():
    with pytest.raises(ValueError, match="MOBILERUN_DEVICE_TOKEN"):
        create_android_portal_http_driver(PORTAL_URL, None)


def test_portal_http_driver_normalizes_url():
    driver = create_android_portal_http_driver(PORTAL_URL + "/", "sekret")

    assert isinstance(driver, AndroidPortalHttpDriver)
    assert driver.url == PORTAL_URL
    assert driver.token == "sekret"


@pytest.fixture
def no_adb(monkeypatch):
    from async_adbutils import adb

    async def fail(*args, **kwargs):
        raise AssertionError("ADB must not be used for a Portal HTTP URL")

    monkeypatch.delenv("ANDROID_SERIAL", raising=False)
    monkeypatch.setattr(adb, "list", fail)
    monkeypatch.setattr(adb, "device", fail)


class FakePortalHttpDriver:
    platform = "Android"

    def __init__(self, connect_error=None):
        self.connect_error = connect_error
        self.connected = False

    async def connect(self):
        if self.connect_error:
            raise self.connect_error
        self.connected = True

    async def get_ui_tree(self):
        return {}


def _record_factory(monkeypatch, module, driver):
    created = {}

    def factory(url, token):
        created["url"] = url
        created["token"] = token
        if not token:
            return create_android_portal_http_driver(url, token)
        return driver

    monkeypatch.setattr(module, "create_android_portal_http_driver", factory)
    return created


class FakeContext:
    def __init__(self):
        self.store = self

    def write_event_to_stream(self, event):
        return None

    async def set(self, key, value):
        setattr(self, key, value)

    async def get(self, key, default=None):
        return getattr(self, key, default)


class ConnectFailed(Exception):
    pass


def _agent_module(monkeypatch):
    from mobilerun.agents.workflows.droid import droid_agent as agent_module

    def unexpected(*args, **kwargs):
        raise AssertionError("ADB-backed setup must not run for a Portal HTTP URL")

    monkeypatch.setattr(agent_module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(agent_module, "capture", lambda *a, **kw: None)
    monkeypatch.setattr(agent_module, "AndroidDevice", unexpected)
    monkeypatch.setattr(agent_module, "ensure_portal_ready", unexpected)
    return agent_module


def test_agent_routes_url_to_portal_http_driver(monkeypatch, no_adb):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch)
    created = _record_factory(monkeypatch, agent_module, FakePortalHttpDriver(ConnectFailed()))
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "sekret")

    config = MobileConfig(
        device=DeviceConfig(serial=PORTAL_URL),
        telemetry=TelemetryConfig(enabled=False),
    )
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())

    with pytest.raises(ConnectFailed):
        run(agent.start_handler(FakeContext(), StartEvent()))

    assert created == {"url": PORTAL_URL, "token": "sekret"}
    assert agent.android_serial is None


def test_external_agent_rejects_portal_http_url(monkeypatch, no_adb):
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch)

    async def fail_run(**kwargs):
        raise AssertionError("external agent must not run")

    monkeypatch.setattr(agent_module, "load_agent", lambda _name: {"config": {}, "run": fail_run})

    config = MobileConfig(
        device=DeviceConfig(serial=PORTAL_URL),
        telemetry=TelemetryConfig(enabled=False),
    )
    config.agent.name = "external-agent"
    agent = agent_module.MobileAgent("Open settings", config=config)

    with pytest.raises(ValueError, match="External agents need an ADB device"):
        run(agent.start_handler(FakeContext(), StartEvent()))


def test_agent_treats_device_date_as_optional(monkeypatch, no_adb):
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core.workflow import StartEvent

    agent_module = _agent_module(monkeypatch)

    class NoDateDriver(FakePortalHttpDriver):
        supported = {"tap", "swipe", "input_text", "screenshot", "_get_ui_tree"}
        supported_buttons = {"back", "home"}

        async def get_date(self):
            raise NotImplementedError

    _record_factory(monkeypatch, agent_module, NoDateDriver())
    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "sekret")

    config = MobileConfig(
        device=DeviceConfig(serial=PORTAL_URL),
        telemetry=TelemetryConfig(enabled=False),
    )
    config.logging.save_trajectory = "none"
    agent = agent_module.MobileAgent("Open settings", config=config, llms=MockLLM())

    run(agent.start_handler(FakeContext(), StartEvent()))

    assert agent.shared_state.device_date == ""
    assert isinstance(agent.driver._driver, NoDateDriver)


def test_external_agent_uses_injected_android_serial(monkeypatch):
    from llama_index.core.workflow import StartEvent
    from mobilerun.core.devices.android import AndroidDevice

    from mobilerun.agents.workflows.droid import droid_agent as module

    monkeypatch.setattr(module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_agent", lambda name: {"config": {}, "run": object()})
    seen = []

    async def device(*, serial):
        seen.append(serial)
        return object()

    async def run_agent(*args, **kwargs):
        return {"success": True, "reason": "done"}

    monkeypatch.setattr(module.adb, "device", device)
    monkeypatch.setattr(module, "run_with_context", run_agent)
    config = MobileConfig(
        device=DeviceConfig(serial="stale"), telemetry=TelemetryConfig(enabled=False)
    )
    config.agent.name = "external"
    agent = module.MobileAgent("go", config=config, driver=AndroidDevice(serial="emulator-5554"))
    run(agent.start_handler(FakeContext(), StartEvent()))
    assert seen == ["emulator-5554"]


def test_external_agent_rejects_incompatible_injected_device(monkeypatch):
    from llama_index.core.workflow import StartEvent
    from mobilerun.core.devices.remote import RemoteDevice

    from mobilerun.agents.workflows.droid import droid_agent as module

    monkeypatch.setattr(module, "setup_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(module, "load_agent", lambda name: {"config": {}, "run": object()})
    config = MobileConfig(telemetry=TelemetryConfig(enabled=False))
    config.agent.name = "external"
    agent = module.MobileAgent(
        "go", config=config, driver=RemoteDevice(url="http://localhost:1234")
    )
    with pytest.raises(ValueError, match="require an injected AndroidDevice backed by ADB"):
        run(agent.start_handler(FakeContext(), StartEvent()))

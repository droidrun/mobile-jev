from importlib import metadata
from unittest.mock import Mock

import pytest
from mobilerun.core.operator import Operator
from mobilerun.models.results import ExecutionResult
from pydantic import BaseModel

from mobilerun_agents import ExternalOperator
from tests.unit.device_fixtures import MemoryDevice


class Count(BaseModel):
    items: int


class InventoryOperator(Operator):
    def __init__(self):
        self.calls = []
        self.result = ExecutionResult[Count](success=True, output=Count(items=3), device="demo")

    async def execute(self, device, instruction, *, output=None, max_steps=None, timeout=None):
        self.calls.append((device, instruction, output, max_steps, timeout))
        return self.result


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("export_class", [True, False])
async def test_external_operator_loads_once_and_returns_plugin_result(monkeypatch, export_class):
    plugins = []

    class Plugin(InventoryOperator):
        def __init__(self):
            super().__init__()
            plugins.append(self)

    entry = Mock(spec=metadata.EntryPoint)
    entry.load.return_value = Plugin if export_class else Plugin()
    discover = Mock(return_value=[entry])
    monkeypatch.setattr(metadata, "entry_points", discover)

    operator = ExternalOperator("inventory")
    discover.assert_not_called()
    device = MemoryDevice()
    first = await operator.execute(device, "Count items", output=Count, max_steps=4, timeout=2.5)
    second = await operator.execute(device, "Count again")
    [plugin] = plugins
    assert first is second is plugin.result
    discover.assert_called_once_with(group="mobilerun.agents", name="inventory")
    entry.load.assert_called_once_with()
    assert plugin.calls == [
        (device, "Count items", Count, 4, 2.5),
        (device, "Count again", None, None, None),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("export", [object(), object])
async def test_external_operator_rejects_non_operator_exports(monkeypatch, export):
    entry = Mock(spec=metadata.EntryPoint)
    entry.load.return_value = export
    monkeypatch.setattr(metadata, "entry_points", lambda **filters: [entry])
    with pytest.raises(ValueError, match=r"inventory.*Operator"):
        await ExternalOperator("inventory").execute(MemoryDevice(), "Count items")


@pytest.mark.anyio
@pytest.mark.parametrize("count", [0, 2])
async def test_external_operator_requires_exactly_one_entry_point(monkeypatch, count):
    entry = Mock(spec=metadata.EntryPoint)
    monkeypatch.setattr(metadata, "entry_points", lambda **filters: [entry] * count)
    with pytest.raises(ValueError, match=rf"inventory.*found {count}"):
        await ExternalOperator("inventory").execute(MemoryDevice(), "Count items")
    entry.load.assert_not_called()


@pytest.mark.anyio
async def test_device_do_honors_external_operator_device_check(monkeypatch):
    device = MemoryDevice()

    class Plugin(InventoryOperator):
        def check(self, checked_device):
            assert checked_device is device
            raise ValueError("unsupported fixture device")

    plugin = Plugin()
    entry = Mock(spec=metadata.EntryPoint)
    entry.load.return_value = plugin
    monkeypatch.setattr(metadata, "entry_points", lambda **filters: [entry])

    with pytest.raises(ValueError, match="unsupported fixture device"):
        await device.do("Count items", operator=ExternalOperator("inventory"))
    assert not plugin.calls

"""Test script-level behavior not exercised by the demo catalog tests."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "demo/demo.py"


def test_entrypoint_reports_keyboard_interrupt_and_exits_130(monkeypatch, capsys):
    demo = runpy.run_path(str(SCRIPT))

    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    asyncio_run = MagicMock(side_effect=interrupt)
    monkeypatch.setitem(demo["run_main"].__globals__, "asyncio", SimpleNamespace(run=asyncio_run))

    with pytest.raises(SystemExit) as raised:
        demo["run_main"]()

    assert raised.value.code == 130
    assert capsys.readouterr().out.strip() == "✗ cancelled"
    asyncio_run.assert_called_once()

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_reports_missing_and_accepts_complete_set(capsys):
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/check_base_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    modules = {"mobilerun.foo", "mobilerun.bar"}
    assert mod.check_modules(modules, lambda name: name == "mobilerun.foo", "0.7.4") == 1
    output = capsys.readouterr().err
    assert "mobilerun 0.7.4 lacks required APIs" in output
    assert "mobilerun.bar" in output
    assert "base rework release" in output
    assert mod.check_modules(modules, lambda name: True, "future") == 0

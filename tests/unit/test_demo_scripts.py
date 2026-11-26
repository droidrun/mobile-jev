"""Exercise the demo runner without a device or provider connection."""

import asyncio
import inspect
import json
import os
import py_compile
import runpy
import signal
import socket
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "demo/demo.py"
NAMES = (
    "easy_jev_executor easy_llm_executor tiktok basic compare_models jev_only jev_exclude jev_delegate jev_executor no_vision vision vision_only "
    "jev_no_vision jev_vision jev_vision_only compare_setups planning structured custom_tool "
    "macro device credentials phone_number"
).split()


def test_editable_runner_compiles(tmp_path):
    assert SCRIPT.is_file()
    py_compile.compile(str(SCRIPT), cfile=str(tmp_path / "demo.pyc"), doraise=True)
    assert "# --- edit me" in SCRIPT.read_text()


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


def test_one_sigint_cancels_planning_child_and_bounds_slow_device_close(monkeypatch, capsys):
    from mobilerun_agents import PlanningOperator
    from mobilerun_agents.agent.action_selector.base import ActionSelector
    from mobilerun_agents.agent.analytics.consumer import RunAnalytics
    from mobilerun_agents.agent.planner import Plan
    from mobilerun_agents.config.loader import ConfigLoader
    from mobilerun_agents.config.user_config import MobileConfig
    from tests.unit.agent.workflows.test_planning import ScriptedPlanner
    from tests.unit.agent.workflows.test_run import resources

    events = []

    class Waiting(ActionSelector):
        async def select(self, run):
            try:
                asyncio.get_running_loop().call_soon(signal.raise_signal, signal.SIGINT)
                await asyncio.Future()
            finally:
                events.append("child cancelled")

    class SlowConnection:
        async def __aenter__(self):
            return (await resources())["device"]

        async def __aexit__(self, exc_type, exc, tb):
            await self.close()

        async def close(self):
            events.append("close started")
            try:
                await asyncio.sleep(3)
            finally:
                events.append("close finished")

    async def flush(*args):
        events.append("telemetry flush started")
        await asyncio.sleep(3)

    async def demo_case(device):
        operator = PlanningOperator(ScriptedPlanner(Plan(("Wait",))), Waiting())
        return await operator.execute(device, "Wait")

    demo = runpy.run_path(str(SCRIPT))
    namespace = demo["main"].__globals__
    monkeypatch.setitem(namespace, "CloudDevice", lambda **kwargs: SlowConnection())
    monkeypatch.setitem(namespace, "CLOUD", True)
    monkeypatch.setitem(namespace, "RESET", False)
    monkeypatch.setitem(namespace, "RUN", ["fake"])
    monkeypatch.setitem(namespace, "DEMOS", {"fake": demo_case})
    monkeypatch.setattr(RunAnalytics, "finish", flush)
    config = MobileConfig()
    config.analytics.enabled = config.checkpoints.enabled = config.tracing.enabled = False
    original = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        with ConfigLoader.override(config), pytest.raises(SystemExit) as raised:
            started = monotonic()
            demo["run_main"]()
    finally:
        signal.signal(signal.SIGINT, original)

    assert raised.value.code == 130
    assert capsys.readouterr().out.count("✗ cancelled") == 1
    assert events == ["child cancelled", "close started", "close finished"]
    assert monotonic() - started < 2


def test_import_is_idle_and_every_selection_is_runnable(monkeypatch, capsys):
    import mobilerun

    def unexpected(*args, **kwargs):
        pytest.fail("import started a demo, opened a device, or connected to the network")

    monkeypatch.setattr(asyncio, "run", unexpected)
    monkeypatch.setattr(socket.socket, "connect", unexpected)
    for name in ("AndroidDevice", "CloudDevice", "RemoteDevice"):
        monkeypatch.setattr(mobilerun, name, unexpected)
    demo = runpy.run_path(str(SCRIPT))
    for name in [*demo["RUN"], *demo["SETUPS"]]:
        assert name in demo["DEMOS"] or name == "all"
    assert set(demo["DEMOS"]) == set(NAMES)
    for name, fn in demo["DEMOS"].items():
        assert inspect.iscoroutinefunction(fn)
        assert fn is demo[name]
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_failure_is_safe_and_next_demo_reuses_device(monkeypatch, capsys):
    main = runpy.run_path(str(SCRIPT))["main"]
    namespace = main.__globals__
    secret = "demo-test-credential"
    monkeypatch.setenv("TYPESAFE_API_KEY", secret)
    connection = MagicMock()
    device = connection.__aenter__.return_value
    device.press_button = AsyncMock()
    device.screenshot = AsyncMock(return_value=b"image")
    device.state.refresh = AsyncMock(return_value=SimpleNamespace(elements=[{}, {}]))
    device.list_apps = AsyncMock(return_value=[{}])
    cloud = MagicMock(return_value=connection)
    monkeypatch.setitem(namespace, "CloudDevice", cloud)
    monkeypatch.setitem(namespace, "CLOUD", True)
    monkeypatch.setitem(namespace, "RESET", False)
    monkeypatch.setitem(namespace, "RUN", ["jev_delegate", "device"])
    monkeypatch.setitem(namespace, "DecisionSelector", MagicMock())
    llm = AsyncMock(side_effect=RuntimeError(f"provider unavailable: {secret}"))
    monkeypatch.setitem(namespace, "build_llm", llm)

    asyncio.run(main())

    text = capsys.readouterr().out
    assert secret not in text
    assert "RuntimeError: provider unavailable: ***" in text
    assert "5 bytes" in text and "2 UI elements" in text and "1 apps" in text
    llm.assert_awaited_once()
    cloud.assert_called_once()
    connection.__aexit__.assert_awaited_once()


@pytest.mark.parametrize("debug", [False, True])
def test_main_configures_cli_logging_once(monkeypatch, debug):
    from mobilerun_agents.agent import logs

    configure = MagicMock()
    monkeypatch.setattr(logs, "configure_cli_logging", configure)
    main = runpy.run_path(str(SCRIPT))["main"]
    monkeypatch.setitem(main.__globals__, "DEBUG", debug)
    monkeypatch.setitem(main.__globals__, "CloudDevice", MagicMock())
    monkeypatch.setitem(main.__globals__, "RUN", ["basic", "device"])
    monkeypatch.setitem(main.__globals__, "RESET", False)
    monkeypatch.setitem(
        main.__globals__,
        "DEMOS",
        {name: AsyncMock(return_value="ok") for name in ["basic", "device"]},
    )
    asyncio.run(main())
    configure.assert_called_once_with(debug=debug)


def test_device_reports_partial_inventory_on_remote():
    from mobilerun.errors import DeviceOperationNotSupported

    device = MagicMock()
    device.screenshot = AsyncMock(return_value=b"image")
    device.state.refresh = AsyncMock(return_value=SimpleNamespace(elements=[]))
    device.list_apps = AsyncMock(side_effect=DeviceOperationNotSupported("No app inventory"))
    result = asyncio.run(runpy.run_path(str(SCRIPT))["device"](device))
    assert "5 bytes" in result and "app list unavailable" in result


def test_comparison_keeps_failed_provider_in_table_and_continues(monkeypatch, capsys):
    compare = runpy.run_path(str(SCRIPT))["compare_models"]
    choose = AsyncMock(side_effect=[ValueError("model unavailable"), MagicMock()])
    monkeypatch.setitem(compare.__globals__, "COMPARE", ["openai", "anthropic"])
    monkeypatch.setitem(compare.__globals__, "MODELS", {"openai": "m1", "anthropic": "m2"})
    monkeypatch.setitem(compare.__globals__, "RESET", False)
    monkeypatch.setitem(compare.__globals__, "selector", choose)
    run = AsyncMock(return_value=SimpleNamespace(success=True, steps=2, answer="Battery\n80%"))
    monkeypatch.setitem(compare.__globals__, "execute", run)
    asyncio.run(compare(MagicMock()))
    rows = [r.split("\t") for r in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["provider/model", "success", "steps", "seconds", "answer"]
    assert rows[1][:3] == ["openai/m1", "False", "-"]
    assert rows[1][4] == "ValueError: model unavailable"
    assert rows[2][:3] == ["anthropic/m2", "True", "2"]
    assert rows[2][4] == "Battery 80%"


@pytest.fixture
def demo_wiring(monkeypatch):
    from mobilerun.models.results import ExecutionResult

    demo = runpy.run_path(str(SCRIPT))
    namespace = demo["main"].__globals__
    fakes = SimpleNamespace(
        AgentOperator=MagicMock(),
        PlanningOperator=MagicMock(),
        DecisionSelector=MagicMock(),
        TypeSafeDecisionModel=MagicMock(),
        DelegatingSelector=MagicMock(),
        LlamaIndexLLMSelector=MagicMock(),
        LlamaIndexPlanner=MagicMock(),
        build_llm=AsyncMock(),
        execute=AsyncMock(return_value=ExecutionResult(success=True, device="fake")),
    )
    for name, fake in vars(fakes).items():
        monkeypatch.setitem(namespace, name, fake)
    return namespace, fakes


@pytest.mark.parametrize(
    ("name", "mode"), [("no_vision", "off"), ("vision", "on"), ("vision_only", "only")]
)
def test_llm_modes_use_the_corresponding_vision_and_model(demo_wiring, monkeypatch, name, mode):
    demo, fakes = demo_wiring
    monkeypatch.setitem(demo, "VISION_MODEL", ("gemini", "vision-model"))
    asyncio.run(demo["DEMOS"][name](MagicMock()))
    fakes.build_llm.assert_awaited_once_with("gemini", "vision-model")
    fakes.LlamaIndexLLMSelector.assert_called_once_with(fakes.build_llm.return_value, vision=mode)
    fakes.AgentOperator.assert_called_once_with(selector=fakes.LlamaIndexLLMSelector.return_value)


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("jev_only", None),
        ("jev_no_vision", "off"),
        ("jev_vision", "on"),
        ("jev_vision_only", "only"),
    ],
)
def test_jev_modes_do_not_construct_an_llm(demo_wiring, name, mode):
    demo, fakes = demo_wiring
    asyncio.run(demo["DEMOS"][name](MagicMock()))
    kwargs = {} if mode is None else {"vision": mode}
    fakes.DecisionSelector.assert_called_once_with(
        fakes.TypeSafeDecisionModel.return_value, **kwargs
    )
    fakes.AgentOperator.assert_called_once_with(selector=fakes.DecisionSelector.return_value)
    fakes.build_llm.assert_not_awaited()


def test_jev_exclude_setup_omits_press_button(demo_wiring):
    demo, fakes = demo_wiring
    asyncio.run(demo["DEMOS"]["jev_exclude"](MagicMock()))
    fakes.DecisionSelector.assert_called_once_with(
        fakes.TypeSafeDecisionModel.return_value, exclude={"press_button"}
    )
    fakes.AgentOperator.assert_called_once_with(selector=fakes.DecisionSelector.return_value)


@pytest.mark.parametrize("name", ["easy_jev_executor", "easy_llm_executor"])
def test_easy_executor_demos_use_easy_settings(demo_wiring, monkeypatch, name):
    demo, fakes = demo_wiring
    monkeypatch.setitem(demo, "EASY_GOAL", "easy task")
    monkeypatch.setitem(demo, "EASY_JEV_VISION", "on")
    monkeypatch.setitem(demo, "EASY_JEV_MIN_CONFIDENCE", 0.42)
    monkeypatch.setitem(demo, "EASY_MAX_STEPS", 9)
    monkeypatch.setitem(demo, "PLANNER", ("planner", "p"))
    monkeypatch.setitem(demo, "EXECUTOR", ("executor", "e"))
    asyncio.run(demo["DEMOS"][name](MagicMock()))
    fakes.build_llm.assert_has_awaits([call("planner", "p"), call("executor", "e")])
    if name == "easy_jev_executor":
        fakes.DecisionSelector.assert_called_once_with(
            fakes.TypeSafeDecisionModel.return_value, vision="on"
        )
        fakes.DelegatingSelector.assert_called_once_with(
            fakes.DecisionSelector.return_value,
            fakes.LlamaIndexLLMSelector.return_value,
            min_confidence=0.42,
        )
        assert fakes.LlamaIndexLLMSelector.call_args.kwargs == {"vision": "on"}
        executor = fakes.DelegatingSelector.return_value
    else:
        assert fakes.LlamaIndexLLMSelector.call_args.kwargs == {"vision": "on"}
        executor = fakes.LlamaIndexLLMSelector.return_value
    fakes.PlanningOperator.assert_called_once_with(
        planner=fakes.LlamaIndexPlanner.return_value, executor=executor
    )
    fakes.execute.assert_awaited_once_with(
        fakes.PlanningOperator.return_value, ANY, "easy task", max_steps=9
    )


def test_tiktok_uses_and_closes_separate_device_even_on_error(demo_wiring, monkeypatch):
    demo, fakes = demo_wiring
    connection = MagicMock()
    phone = connection.__aenter__.return_value
    monkeypatch.setitem(demo, "TIKTOK_DEVICE", "separate")
    monkeypatch.setitem(demo, "TIKTOK_CLOUD", True)
    monkeypatch.setitem(demo, "RESET", True)
    monkeypatch.setitem(demo, "CloudDevice", MagicMock(return_value=connection))
    run = AsyncMock(side_effect=RuntimeError("failure"))
    monkeypatch.setitem(demo, "execute", run)
    result = asyncio.run(demo["tiktok"](MagicMock()))
    assert result == "failed: RuntimeError"
    demo["CloudDevice"].assert_called_once_with(device_id="separate")
    phone.press_button.assert_not_called()
    connection.__aexit__.assert_awaited_once()
    fakes.AgentOperator.assert_called_once_with(selector=fakes.LlamaIndexLLMSelector.return_value)
    assert run.call_args.args[1] is phone
    assert run.call_args.args[2] == demo["TIKTOK_GOAL"]
    assert run.call_args.kwargs["max_steps"] == demo["TIKTOK_MAX_STEPS"]


def test_tiktok_skips_without_device(demo_wiring, monkeypatch, capsys):
    demo, _ = demo_wiring
    monkeypatch.setitem(demo, "TIKTOK_DEVICE", None)
    assert asyncio.run(demo["tiktok"](MagicMock())) == "skipped: set TIKTOK_DEVICE"


@pytest.mark.parametrize("name", ["jev_delegate", "jev_executor"])
def test_delegation_wires_independent_vision_and_jev_is_only_an_executor(
    demo_wiring, monkeypatch, name
):
    demo, fakes = demo_wiring
    monkeypatch.setitem(demo, "JEV_MIN_CONFIDENCE", 0.6)
    monkeypatch.setitem(demo, "JEV_VISION", "on")
    monkeypatch.setitem(demo, "DELEGATE_VISION", "off")
    asyncio.run(demo["DEMOS"][name](MagicMock()))
    fakes.DecisionSelector.assert_called_once_with(
        fakes.TypeSafeDecisionModel.return_value, vision="on"
    )
    llm_selector = call(fakes.build_llm.return_value, vision="off")
    assert llm_selector in fakes.LlamaIndexLLMSelector.call_args_list
    fakes.DelegatingSelector.assert_called_once_with(
        fakes.DecisionSelector.return_value,
        fakes.LlamaIndexLLMSelector.return_value,
        min_confidence=0.6,
    )
    if name == "jev_executor":
        fakes.LlamaIndexPlanner.assert_called_once_with(fakes.build_llm.return_value)
        fakes.PlanningOperator.assert_called_once_with(
            planner=fakes.LlamaIndexPlanner.return_value,
            executor=fakes.DelegatingSelector.return_value,
        )
    else:
        fakes.AgentOperator.assert_called_once_with(selector=fakes.DelegatingSelector.return_value)


def test_compare_setups_keeps_failed_row_and_runs_same_goal(demo_wiring, monkeypatch, capsys):
    from mobilerun.models.results import ExecutionResult

    demo, fakes = demo_wiring
    monkeypatch.setitem(demo, "SETUPS", ["unknown", "jev_only", "no_vision", "vision"])
    monkeypatch.setitem(demo, "GOAL", "Check test battery")
    monkeypatch.setitem(demo, "RESET", False)
    fakes.execute.side_effect = [
        RuntimeError("unavailable"),
        ExecutionResult(success=True, steps=2, answer="Battery\n80%", device="fake"),
        ExecutionResult(success=False, steps=3, answer="No result", device="fake"),
    ]
    phone = MagicMock()
    assert asyncio.run(demo["compare_setups"](phone)) == "Compared 4 setup(s)."
    rows = [r.split("\t") for r in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["setup", "success", "steps", "seconds", "tokens", "answer"]
    assert [r[:3] for r in rows[1:]] == [
        ["unknown", "False", "-"],
        ["jev_only", "False", "-"],
        ["no_vision", "True", "2"],
        ["vision", "False", "3"],
    ]
    assert [r[4:] for r in rows[1:]] == [
        ["-", "Unknown setup"],
        ["-", "RuntimeError: unavailable"],
        ["-", "Battery 80%"],
        ["-", "No result"],
    ]
    assert all(float(r[3]) >= 0 for r in rows[1:])
    assert (
        fakes.execute.await_args_list
        == [
            call(
                fakes.AgentOperator.return_value,
                phone,
                "Check test battery",
            )
        ]
        * 3
    )


def test_jev_accepts_a_saved_key_without_environment(demo_wiring, monkeypatch, capsys):
    from mobilerun_agents.agent.auth import resolve, store

    demo, fakes = demo_wiring
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    key = "fake-saved-jev-credential"
    store.save_key("typesafe", key)

    async def execute(*args):
        assert (await resolve("typesafe")).api_key == key
        return fakes.execute.return_value

    fakes.execute.side_effect = execute
    assert asyncio.run(demo["jev_only"](MagicMock())).success
    assert key not in capsys.readouterr().out


def test_compare_models_skips_missing_keys_and_runs_saved_and_env_providers(
    demo_wiring, monkeypatch, capsys
):
    from mobilerun_agents.agent.auth import resolve, store

    demo, fakes = demo_wiring
    store.save_key("anthropic", "fake-saved-llm-credential")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-env-llm-credential")
    monkeypatch.setitem(demo, "COMPARE", ["openai", "anthropic", "gemini"])
    monkeypatch.setitem(demo, "MODELS", {"openai": "o", "anthropic": "a", "gemini": "g"})
    monkeypatch.setitem(demo, "RESET", False)

    async def build_llm(provider, model):
        return await resolve(provider)

    fakes.build_llm.side_effect = build_llm
    asyncio.run(demo["compare_models"](MagicMock()))

    output = capsys.readouterr().out
    rows = [line.split("\t") for line in output.splitlines()]
    assert rows[1][:3] == ["openai/o", "-", "-"]
    assert rows[1][-1] == "skipped (no key)"
    assert [row[:2] for row in rows[2:]] == [["anthropic/a", "True"], ["gemini/g", "True"]]
    assert "fake-saved-llm-credential" not in output and "fake-env-llm-credential" not in output
    assert fakes.execute.await_count == 2


@pytest.mark.parametrize("override", [False, True])
def test_macro_chooses_newest_recording_and_respects_explicit_path(
    demo_wiring, monkeypatch, tmp_path, override
):
    demo, fakes = demo_wiring
    monkeypatch.chdir(tmp_path)
    if override:
        directory = tmp_path / "home" / "recorded-macros"
        monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", "~/recorded-macros")
    else:
        directory = tmp_path / "config" / "macros"
    monkeypatch.setitem(demo, "MACRO", None)
    for name, mtime in [("zzz-old", 1), ("aaa-new", 2)]:
        path = directory / name / "macro.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
        os.utime(path, (mtime, mtime))
    (directory / "incomplete" / "macro.json").mkdir(parents=True)
    macro = MagicMock()
    monkeypatch.setitem(demo, "MacroAgentOperator", macro)

    asyncio.run(demo["macro"](MagicMock()))
    macro.assert_called_once_with(
        macro_file=directory / "aaa-new" / "macro.json", agent=fakes.AgentOperator.return_value
    )
    monkeypatch.setitem(demo, "MACRO", "my-run")
    asyncio.run(demo["macro"](MagicMock()))
    assert macro.call_args.kwargs["macro_file"] == "my-run"


def test_macro_without_recordings_skips_before_building_llm(demo_wiring, monkeypatch, tmp_path):
    demo, fakes = demo_wiring
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "empty-macros"))
    monkeypatch.setitem(demo, "MACRO", None)
    assert (
        asyncio.run(demo["macro"](MagicMock()))
        == "skipped: no recorded macro yet (run any demo first)"
    )
    fakes.build_llm.assert_not_awaited()
    fakes.execute.assert_not_awaited()


def test_compare_setups_all_uses_only_comparable_demos(demo_wiring, monkeypatch, capsys):
    demo, _ = demo_wiring
    monkeypatch.setitem(demo, "SETUPS", ["all"])
    monkeypatch.setitem(demo, "RESET", False)

    asyncio.run(demo["compare_setups"](MagicMock()))

    rows = capsys.readouterr().out.splitlines()[1:]
    assert [row.split("\t")[0] for row in rows] == list(demo["SETUP_DEMOS"])


@pytest.mark.parametrize("reset", [True, False])
def test_main_and_both_comparisons_reset_before_each_run(demo_wiring, monkeypatch, reset):
    from mobilerun.models.results import ExecutionResult

    demo, fakes = demo_wiring
    events = []
    cloud = MagicMock()
    phone = cloud.return_value.__aenter__.return_value

    async def press(button):
        events.append(button)

    async def wait(seconds):
        assert 0 < seconds <= 2
        events.append("wait")

    async def execute(*args):
        events.append("run")
        return ExecutionResult(success=True, device="fake")

    phone.press_button = AsyncMock(side_effect=press)
    fakes.execute.side_effect = execute
    monkeypatch.setitem(demo, "sleep", wait)
    monkeypatch.setitem(demo, "CloudDevice", cloud)
    monkeypatch.setitem(demo, "CLOUD", True)
    monkeypatch.setitem(demo, "RESET", reset)
    monkeypatch.setitem(demo, "RUN", ["basic", "compare_models", "compare_setups"])
    monkeypatch.setitem(demo, "DEFAULT", "openai")
    monkeypatch.setitem(demo, "COMPARE", ["openai", "anthropic"])
    monkeypatch.setitem(demo, "MODELS", {"openai": "o", "anthropic": "a"})
    monkeypatch.setitem(demo, "SETUPS", ["jev_only", "no_vision"])

    asyncio.run(demo["main"]())

    move = ["home", "wait"] if reset else []
    assert events == (move + ["run"]) + (move + (move + ["run"]) * 2) * 2


@pytest.mark.parametrize("demo_name", ["credentials", "phone_number"])
@pytest.mark.parametrize("failure", [None, "bind", "run"])
def test_extension_binding_is_temporary(demo_wiring, monkeypatch, capsys, demo_name, failure):
    demo, fakes = demo_wiring
    device = MagicMock()
    device.supported_operations = {"cloud_telephony"}
    calls = []
    fake_secret = "fixture-secret-not-for-output"

    class Extension:
        device_id = None

        def __await__(self):
            return ready().__await__()

        async def bind(self, phone):
            assert phone is device
            calls.append("bind")
            if failure == "bind":
                raise RuntimeError(fake_secret)
            return self

        async def unbind(self):
            calls.append("unbind")

        async def close(self):
            calls.append("close")

    async def ready():
        return extension

    async def execute(operator, phone, goal):
        assert phone is device
        assert fake_secret not in goal
        config = demo["ConfigLoader"].load()
        assert config.checkpoints.enabled
        if demo_name == "phone_number":
            assert "sms.send" in config.tools.exclude
        calls.append("run")
        if failure == "run":
            raise RuntimeError(fake_secret)
        return fakes.execute.return_value.model_copy(update={"answer": fake_secret})

    extension = Extension()
    provider = MagicMock(return_value=extension)
    monkeypatch.setitem(demo, "DictCredentials", provider)
    monkeypatch.setitem(demo, "CloudPhoneNumber", provider)
    monkeypatch.setitem(demo, "PHONE_NUMBER_ID", "existing-number")
    monkeypatch.setitem(demo, "CREDENTIAL_PASSWORD", fake_secret)
    monkeypatch.setitem(demo, "execute", execute)

    result = asyncio.run(demo[demo_name](device))

    assert calls == ["bind"] + ([] if failure == "bind" else ["run"]) + ["unbind", "close"]
    assert isinstance(result, str) and fake_secret not in result
    capture = capsys.readouterr()
    assert fake_secret not in capture.out and fake_secret not in capture.err
    if demo_name == "credentials":
        assert demo["CREDENTIAL_PACKAGE"] == "com.android.chrome"
        provider.assert_called_once_with(
            {
                demo["CREDENTIAL_PACKAGE"]: {
                    demo["CREDENTIAL_NAME"]: {
                        "password": fake_secret,
                    }
                }
            }
        )
    else:
        provider.assert_called_once_with("existing-number")


def test_cloud_credentials_uses_existing_name_without_saving(demo_wiring, monkeypatch):
    demo, fakes = demo_wiring
    provider = MagicMock()
    provider.return_value.bind = AsyncMock()
    provider.return_value.unbind = AsyncMock()
    provider.return_value.close = AsyncMock()
    monkeypatch.setitem(demo, "CloudCredentials", provider)
    monkeypatch.setitem(demo, "CREDENTIAL_SOURCE", "cloud")
    monkeypatch.setitem(demo, "CREDENTIAL_NAME", "existing-qa-entry")

    asyncio.run(demo["credentials"](MagicMock()))

    assert "existing-qa-entry" in fakes.execute.call_args.args[2]
    assert [action[0] for action in provider.return_value.method_calls] == [
        "bind",
        "unbind",
        "close",
    ]


def test_default_credential_demo_targets_chrome_search_bar(demo_wiring):
    demo, _ = demo_wiring
    assert demo["CREDENTIAL_PACKAGE"] == "com.android.chrome"
    assert "search bar" in demo["CREDENTIAL_GOAL"]
    assert "{name}" in demo["CREDENTIAL_GOAL"]
    assert "Don't press enter" in demo["CREDENTIAL_GOAL"]
    assert "URL" not in demo["CREDENTIAL_GOAL"]
    assert "tap" in demo["CREDENTIAL_GOAL"].lower()


def test_missing_phone_number_skips_before_device_connection(demo_wiring, monkeypatch, capsys):
    demo, fakes = demo_wiring
    connection = MagicMock()
    monkeypatch.setitem(demo, "CloudDevice", connection)
    monkeypatch.setitem(demo, "CLOUD", True)
    monkeypatch.setitem(demo, "RUN", ["phone_number"])
    monkeypatch.setitem(demo, "PHONE_NUMBER_ID", None)

    asyncio.run(demo["main"]())

    assert "skipped: set PHONE_NUMBER_ID" in capsys.readouterr().out
    connection.assert_not_called()
    fakes.execute.assert_not_awaited()


def test_phone_number_never_replaces_existing_device_binding(demo_wiring, monkeypatch):
    demo, fakes = demo_wiring
    number = MagicMock()
    monkeypatch.setitem(demo, "CloudPhoneNumber", number)
    monkeypatch.setitem(demo, "PHONE_NUMBER_ID", "existing-number")
    phone = SimpleNamespace(
        cloud_id="test-device",
        supported_operations={"cloud_telephony"},
        extensions={"sms": object()},
    )

    assert (
        asyncio.run(demo["phone_number"](phone))
        == "skipped: use a test phone without a bound number"
    )
    number.assert_not_called()
    fakes.execute.assert_not_awaited()


def test_phone_number_skips_number_already_attached_elsewhere(demo_wiring, monkeypatch):
    demo, fakes = demo_wiring
    number = SimpleNamespace(device_id="another-device", close=AsyncMock(), bind=AsyncMock())

    async def retrieve(number_id):
        assert number_id == "existing-number"
        return number

    monkeypatch.setitem(demo, "CloudPhoneNumber", retrieve)
    monkeypatch.setitem(demo, "PHONE_NUMBER_ID", "existing-number")
    phone = SimpleNamespace(
        cloud_id="test-device", supported_operations={"cloud_telephony"}, extensions={}
    )

    assert asyncio.run(demo["phone_number"](phone)) == "skipped: use an unbound cloud number"
    number.bind.assert_not_awaited()
    number.close.assert_awaited_once()
    fakes.execute.assert_not_awaited()


@pytest.mark.parametrize("leak", [None, "nested_binary", "macro"])
@pytest.mark.parametrize("generic_checkpoint_name", [False, True])
def test_extension_evidence_checks_last_checkpoint_and_all_artifacts(
    demo_wiring, monkeypatch, tmp_path, capsys, leak, generic_checkpoint_name
):
    demo, _ = demo_wiring
    secret = "artifact-test-fixture-value"
    folder = tmp_path / "runs" / "run-1"
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "0009.json").write_text('{"actions": [{"name": "older"}]}')
    action = "extension_call" if generic_checkpoint_name else "vault.type_field"
    (checkpoints / "10000.json").write_text(
        json.dumps({"actions": [{"name": "tap"}, {"name": action, "args": {}, "ok": True}]})
    )
    macro_dir = tmp_path / "macros"
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(macro_dir))
    macro_file = macro_dir / folder.name / "macro.json"
    macro_file.parent.mkdir(parents=True)
    macro_file.write_text(
        json.dumps(
            {
                "actions": [{"action_type": "extension_call", "name": "vault.type_field"}],
                "description": secret if leak == "macro" else "test",
            }
        )
    )
    if leak == "nested_binary":
        binary = folder / "screenshots" / "0001.png"
        binary.parent.mkdir()
        binary.write_bytes(b"\xff\x00" + secret.encode())

    demo["report_extension_use"](str(folder), "credential", {"vault.type_field"}, (secret,))

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "used: tap, vault.type_field"
    assert "credential tool called: yes" in output
    assert f"secret visible in run folder: {'yes' if leak else 'no'}" in output
    assert secret not in output

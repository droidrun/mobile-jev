import asyncio
from types import SimpleNamespace

from mobilerun.core.devices.base import Device as DeviceMixin
from mobilerun.macro.recorder import MacroRecorder

from mobilerun.agents.tools import actions
from mobilerun.agents.tools.actions import (
    click,
    open_app,
    type_text,
    wait,
)


class FakeDriver(DeviceMixin):
    @property
    def actions(self):
        async def call(name, arguments, *, audience):
            assert audience == "operator"
            if name == "device.wait":
                return None
            return await getattr(self, name.removeprefix("device."))(**arguments)

        return SimpleNamespace(call=call)

    @property
    def _recording(self):
        return self

    @property
    def recorded_actions(self):
        return self.log

    def __init__(self):
        super().__init__()
        self.taps = []
        self.inputs = []
        self.log = []

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.inputs.append((text, clear))
        return True

    async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
        return await self.input_text(text, clear)

    async def start_app(self, package, activity=None):
        self.log.append({"action_type": "start_app", "package": package, "activity": activity})
        return f"App started: {package}"


class FakeUI:
    use_normalized = False
    elements = [
        {
            "index": 7,
            "resourceId": "com.example:id/name",
            "className": "android.widget.EditText",
            "text": "",
            "bounds": "10,20,110,60",
        }
    ]
    phone_state = {"package": "com.example", "activity": ".MainActivity"}
    screen_width = 400
    screen_height = 800

    def get_element_coords(self, index):
        return (60, 40)

    def get_element_info(self, index):
        return {
            "text": "",
            "className": "android.widget.EditText",
            "type": "input",
        }


class SequencedStateProvider:
    def __init__(self, states):
        self.states = list(states)

    async def get_state(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def test_wait_records_first_class_macro_action():
    recorder = MacroRecorder()
    ctx = SimpleNamespace(
        driver=FakeDriver(),
        macro_recorder=recorder,
        ui=FakeUI(),
        state_provider=SequencedStateProvider([FakeUI()]),
    )

    result = asyncio.run(wait(1.25, ctx=ctx))

    assert result.success
    assert "super-secret-password" not in result.summary
    assert recorder.actions[0]["action_type"] == "wait"
    assert recorder.actions[0]["duration"] == 1.25
    assert recorder.actions[0]["pre_state"]["nodes"]


def test_element_click_records_pre_state_without_target_hint():
    recorder = MacroRecorder()
    ctx = SimpleNamespace(
        driver=FakeDriver(),
        ui=FakeUI(),
        macro_recorder=recorder,
        state_provider=SequencedStateProvider([FakeUI()]),
    )

    result = asyncio.run(click(7, ctx=ctx))

    assert result.success
    action = recorder.actions[0]
    assert action["action_type"] == "tap"
    assert action["x"] == 60
    assert action["y"] == 40
    assert action["pre_state"]["nodes"]
    assert "target_hint" not in action


def test_type_text_with_index_records_input_with_refreshed_pre_state():
    first_ui = FakeUI()
    focused_ui = FakeUI()
    focused_ui.elements = [
        {
            "index": 7,
            "resourceId": "com.example:id/name",
            "className": "android.widget.EditText",
            "text": "",
            "focused": True,
            "bounds": "10,20,110,60",
        }
    ]

    recorder = MacroRecorder()
    driver = FakeDriver()
    ctx = SimpleNamespace(
        driver=driver,
        ui=first_ui,
        macro_recorder=recorder,
        state_provider=SequencedStateProvider([first_ui, focused_ui]),
    )

    result = asyncio.run(type_text("hello", index=7, clear=True, ctx=ctx))

    assert result.success
    assert [action["action_type"] for action in recorder.actions] == [
        "tap",
        "input_text",
    ]
    assert recorder.actions[0]["pre_state"]["nodes"][0]["focused"] is None
    assert "target_hint" not in recorder.actions[0]
    assert recorder.actions[1]["pre_state"]["nodes"][0]["focused"] is True
    assert "target_hint" not in recorder.actions[1]


def test_open_app_promotes_recording_driver_start_app_log(monkeypatch):
    class FakeAppStarter:
        def __init__(self, driver, llm, timeout, stream, verbose):
            self._driver = driver

        async def run(self, app_description):
            assert app_description == "Settings"
            return await self._driver.start_app("com.android.settings")

    monkeypatch.setattr(actions, "AppStarter", FakeAppStarter)

    recorder = MacroRecorder()
    driver = FakeDriver()
    ctx = SimpleNamespace(
        driver=driver,
        ui=FakeUI(),
        macro_recorder=recorder,
        app_opener_llm=object(),
        streaming=False,
        state_provider=SequencedStateProvider([FakeUI()]),
    )

    result = asyncio.run(open_app("Settings", ctx=ctx))

    assert result.success
    assert recorder.actions[0]["action_type"] == "start_app"
    assert recorder.actions[0]["package"] == "com.android.settings"
    assert recorder.actions[0]["pre_state"]["nodes"]


def test_sequential_actions_record_refreshed_current_pre_state():
    first_ui = FakeUI()
    second_ui = FakeUI()
    second_ui.elements = [
        {
            "index": 1,
            "resourceId": "com.example:id/after_click",
            "className": "android.widget.TextView",
            "text": "After click",
            "bounds": "0,0,100,100",
        }
    ]

    recorder = MacroRecorder()
    ctx = SimpleNamespace(
        driver=FakeDriver(),
        ui=first_ui,
        macro_recorder=recorder,
        state_provider=SequencedStateProvider([first_ui, second_ui]),
    )

    async def run():
        click_result = await click(7, ctx=ctx)
        wait_result = await wait(0, ctx=ctx)
        return click_result, wait_result

    click_result, wait_result = asyncio.run(run())

    assert click_result.success
    assert wait_result.success
    assert recorder.actions[0]["pre_state"]["nodes"][0]["resource_id"] == ("com.example:id/name")
    assert recorder.actions[1]["pre_state"]["nodes"][0]["resource_id"] == (
        "com.example:id/after_click"
    )

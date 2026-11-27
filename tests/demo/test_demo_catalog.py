"""Exercise the installed operators through the demo runner, offline."""

import asyncio
import base64
import json
import runpy
import socket
from pathlib import Path

import pytest
from llama_index.core.base.llms.types import ImageBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from mobilerun.core.extensions import Extension, action
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import PrivateAttr

from mobilerun_agents.agent.action_selector.decision_model.base import Decision
from mobilerun_agents.agent.action_selector.llm import llama_index
from mobilerun_agents.agent.action_selector.llm.models import Capabilities
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import MobileConfig
from tests.unit.device_fixtures import MemoryDevice

SCRIPT = Path(__file__).with_name("demo.py")
CASES = {
    "llm": "answer=Battery at 80%",
    "llm_vision": "only: success=True",
    "jev": "answer=Jev finished",
    "jev_exclude": "answer=Jev finished",
    "delegation": "answer=Battery at 80%",
    "planner_llm": "answer=Battery at 80%",
    "planner_jev": "answer=No reliable action",
    "planner_delegation": "answer=Battery at 80%",
    "vault_login": "success=True steps=2",
    "phone_number": "not found; steps=2",
    "mailbox": "found; steps=2",
    "structured": '{"level":80,"charging":false}',
    "custom_tool": "answer=Battery at 80%",
    "macro": "replayed 1 action(s)",
    "inspect_run": 'tap_element Button "Battery" | ok=True changed=False',
    "compare_models": "Compared 2 model(s).",
    "compare_setups": "Compared 4 setup(s).",
}


@pytest.fixture
def demo(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("demo attempted a real network connection")

    monkeypatch.setattr(socket.socket, "connect", denied)
    main = runpy.run_path(str(SCRIPT))["main"]
    return main.__globals__


def test_edit_block_is_preserved(demo):
    expected = {
        "DEVICE": "4c1f61fd-5e3a-4d02-9457-478f05875ab9",
        "CLOUD": True,
        "RUN": ["planner_delegation"],
        "DEBUG": False,
        "MAX_STEPS": 60,
        "TIMEOUT": 300,
        "MODELS": {
            "openai": "gpt-6-luna",
            "anthropic": "claude-sonnet-5",
            "gemini": "gemini-3.8-flash",
        },
        "COMPARE": ["openai", "anthropic"],
        "DEFAULT": "openai",
        "PLANNER": ("openai", "gpt-6-luna"),
        "EXECUTOR": ("openai", "gpt-6-luna"),
        "VISION_MODEL": ("openai", "gpt-6-luna"),
        "JEV_VISION": "off",
        "JEV_MIN_CONFIDENCE": 0.7,
        "GOAL": "Check the cheapest prices of Amazon and eBay prices for the game split fiction.",
        "EASY_GOAL": "Open Settings and tell me the battery level.",
        "CREDENTIAL_SOURCE": "local",
        "CREDENTIAL_NAME": "demo-login",
        "CREDENTIAL_PACKAGE": "com.android.chrome",
        "CREDENTIAL_PASSWORD": "test-password-123",
        "CREDENTIAL_GOAL": (
            "Open Chrome, tap the search bar, then call vault.type_field to paste the password "
            "of the saved credential {name} into it. Don't press enter. Report whether the field is filled."
        ),
        "PHONE_NUMBER_ID": None,
        "PHONE_NUMBER_TEST_CODE": None,
        "PHONE_NUMBER_GOAL": (
            "Read the attached phone number and use sms.messages to check the latest inbound SMS "
            "for a verification code. Report whether you found a code, not the code itself."
        ),
    }
    assert {name: demo[name] for name in expected} == expected


def test_catalog_names(demo):
    assert list(demo["DEMOS"]) == list(CASES)


def test_unknown_run_errors_before_connecting(demo, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("opened a device for an unknown example")

    monkeypatch.setitem(demo, "CloudDevice", unexpected)
    monkeypatch.setitem(demo, "RUN", ["llm", "typo"])
    with pytest.raises(ValueError, match="Unknown RUN.*typo.*llm"):
        asyncio.run(demo["main"]())


class Screen(StateProvider):
    capabilities = {"element_index"}
    target_text = "Battery"

    async def get_state(self):
        return UIState(
            [{"index": 1, "text": self.target_text, "className": "Button", "bounds": "1,1,20,20"}],
            "battery",
            "",
            {
                "packageName": "com.android.chrome",
                "focusedElement": {"index": 2},
                "isEditable": True,
            },
            100,
            100,
        )


class Phone(MemoryDevice):
    def __init__(self):
        super().__init__(
            cloud_id="fake",
            supported_operations=frozenset({"press_button", "stop_app", "cloud_telephony", "tap"}),
        )
        self.events = []
        self._set_state_provider(Screen(self))

    async def screenshot(self, hide_overlay=True):
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aUQ0AAAAASUVORK5CYII="
        )

    async def press_button(self, button: str) -> None:
        self.events.append(button)

    async def tap(self, x: int, y: int) -> None:
        self.events.append((x, y))

    async def stop_app(self, package: str) -> str:
        self.events.append(("stop", package))
        return package


class Replies(MockLLM):
    _calls = PrivateAttr(default_factory=list)
    _history = PrivateAttr()
    _inspect = PrivateAttr(default=False)

    async def achat(self, messages, tools, **kwargs):
        names = {tool.metadata.get_name() for tool in tools}
        prompt = " ".join(message.content or "" for message in messages)
        self._history.append(
            (
                prompt,
                names,
                any(
                    isinstance(block, ImageBlock)
                    for message in messages
                    for block in message.blocks
                ),
            )
        )
        tool = "complete"
        args = {"success": True, "message": "Battery at 80%"}
        if "set_plan" in names and not self._calls:
            tool, args = "set_plan", {"subgoals": ["Read battery"]}
        elif "Failure reason: abstained" in prompt and "set_plan" in names:
            args = {"success": False, "message": "No reliable action"}
        elif self._inspect and not self._calls:
            tool, args = "tap_element", {"index": 1}
        elif not self._calls:
            for name, arguments in (
                ("note", {"text": "80%"}),
                (
                    "vault__type_field",
                    {"field": "password", "name": "demo-login", "package": "com.android.chrome"},
                ),
                ("sms__messages", {}),
                ("email__wait_for_code", {"timeout": 1}),
                ("press_button", {"button": "back"}),
            ):
                if name in names:
                    tool, args = name, arguments
                    break
        if tool == "complete" and "charging" in prompt:
            args["output"] = {"level": 80, "charging": False}
        if tool == "complete" and "email__wait_for_code" in names:
            args.update(message="314159", output={"found": True})
        if tool == "complete" and "sms__messages" in names:
            args.update(message="synthetic private SMS body", output={"found": False})
        self._calls.append(tool)
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                blocks=[
                    ToolCallBlock(
                        tool_name=tool, tool_kwargs=args, tool_call_id=f"call-{len(self._calls)}"
                    )
                ],
            )
        )

    async def astream_chat(self, messages, **kwargs):
        async def chunks():
            yield await self.achat(messages, **kwargs)

        return chunks()


class FakeSMS(Extension):
    default_name = "sms"
    device_id = None

    @action()
    async def messages(self) -> list[str]:
        """Read synthetic messages."""
        return []

    @action()
    async def send(self) -> None:
        """Forbidden in the demo."""
        pytest.fail("demo sent a text")


class FakeMail(Extension):
    default_name = "email"

    @action()
    async def wait_for_code(self, timeout: float = 60) -> str:
        """Read a synthetic verification code."""
        return "314159"


@pytest.mark.parametrize("name", CASES)
def test_catalog_runs_with_fakes(demo, monkeypatch, tmp_path, capsys, name):
    phone, history, decisions = Phone(), [], []
    config = MobileConfig()
    config.analytics.enabled = config.tracing.enabled = config.agent.streaming = False
    verification = name in {"phone_number", "mailbox"}
    config.checkpoints.enabled = not verification
    monkeypatch.setitem(demo, "DEBUG", verification)
    monkeypatch.setenv("MOBILERUN_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    monkeypatch.setattr(
        llama_index, "adapter_capabilities", lambda _: Capabilities(tool_call_format="responses")
    )

    async def build_llm(provider, model):
        llm = Replies()
        llm._history = history
        llm._inspect = name == "inspect_run"
        return llm

    class Jev:
        max_options = None
        locates_points = False

        async def decide(self, request):
            options = [option.action.name for option in request.options]
            decisions.append(options)
            return Decision(
                option_index=(
                    options.index("delegate")
                    if "delegate" in options
                    else options.index("complete")
                ),
                confidence=0.2 if name == "planner_jev" else 0.95,
                message="Jev finished",
            )

    async def existing_number(resource_id):
        assert resource_id == "fake-number"
        return FakeSMS()

    async def existing_mailbox(resource_id):
        assert resource_id == "fake-mailbox"
        return FakeMail()

    for key, value in {
        "CloudDevice": lambda **kwargs: phone,
        "build_llm": build_llm,
        "TypeSafeDecisionModel": Jev,
        "CloudPhoneNumber": existing_number,
        "CloudMailbox": existing_mailbox,
        "PHONE_NUMBER_ID": "fake-number",
        "MAILBOX_ID": "fake-mailbox",
        "RUN": ["llm", name] if name in {"macro", "inspect_run"} else [name],
    }.items():
        monkeypatch.setitem(demo, key, value)

    def surface_error(fn):
        async def run(device):
            try:
                return await fn(device)
            except Exception as error:
                pytest.fail(f"{fn.__name__}: {error!r}")

        return run

    monkeypatch.setitem(
        demo, "DEMOS", {key: surface_error(fn) for key, fn in demo["DEMOS"].items()}
    )
    with ConfigLoader.override(config):
        asyncio.run(demo["main"]())

    captured = capsys.readouterr()
    out = captured.out
    assert f"=== {name} ===" in out and CASES[name] in out
    assert not phone.extensions
    assert demo["CREDENTIAL_PASSWORD"] not in out and "314159" not in out
    if verification:
        assert out.strip().splitlines() == [f"=== {name} ===", CASES[name]]
        assert "314159" not in captured.err and "synthetic private SMS body" not in captured.err
        assert not (tmp_path / "runs").exists() and not (tmp_path / "macros").exists()
    if name in {"llm", "llm_vision"}:
        assert {image for _, _, image in history} == ({False} if name == "llm" else {True})
    if name in {"jev", "jev_exclude"}:
        assert not history
    if name == "planner_jev":
        assert len(decisions) == 1
    if name == "jev_exclude":
        assert decisions and all("press_button" not in options for options in decisions)
    if name in {"delegation", "planner_delegation"}:
        assert any("delegate" in options for options in decisions)
    if name == "macro":
        assert phone.events.count("back") == 2
    if name == "vault_login":
        assert phone.inputs == [demo["CREDENTIAL_PASSWORD"]]
    if name == "phone_number":
        assert all("sms__messages" in names and "sms__send" not in names for _, names, _ in history)
    if name in {"compare_models", "compare_setups"}:
        assert out.count("\tTrue\t") == (2 if name == "compare_models" else 4)
    if name == "custom_tool":
        checkpoint = next((tmp_path / "runs").glob("*/context.json"))
        assert "Note saved." in json.dumps(json.loads(checkpoint.read_text()))


def test_reset_stops_foreground_before_home(demo):
    phone = Phone()
    asyncio.run(demo["reset_phone"](phone))
    assert phone.events == [("stop", "com.android.chrome"), "home"]

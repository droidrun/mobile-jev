import importlib.util
from inspect import iscoroutinefunction, signature
from pathlib import Path

import pytest
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import Memory
from llama_index.llms.openai.responses import OpenAIResponses
from llama_index.llms.openai_like import OpenAILike, OpenAILikeResponses
from llama_index.llms.openrouter import OpenRouter
from mobilerun.core.devices import Device
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions.actions import Action, Declaration
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel

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


@pytest.fixture
def ui():
    return UIState([], "", "", {}, 1, 1)


def test_device_cloud_flag():
    # agent/tools/registry.py:_invoke_device
    assert type(Device()._is_cloud) is bool


def test_device_unwrap():
    # agent/tools/registry.py:_invoke_device, _route; agent/logs/redaction.py:observe_vault
    device = Device()
    assert device._unwrap() is device
    assert DeviceWrapper(device)._unwrap() is device


def test_device_input_secret_can_be_wrapped():
    # agent/logs/redaction.py:_track_vault_input
    device = Device()
    original = device._input_secret
    assert iscoroutinefunction(original)
    signature(original).bind("not-a-real-secret")

    async def stub(value: str) -> bool:
        return True

    vars(device)["_input_secret"] = stub
    assert device._input_secret is stub


def test_device_state_cache_is_readable_and_writable(ui):
    # agent/tools/registry.py:_invoke_device, _cached_ui, _target_summary
    view = Device().state
    assert view._current is None
    view._current = ui
    assert view.current is ui


def test_device_can_resolve_state_provider():
    # agent/tools/registry.py:_invoke_device, build_registry
    # agent/tools/atomic/exceptions.py:_dimensions
    # agent/action_selector/{llm,decision_model}/base.py: select
    view = Device(state_provider_factory=lambda device, _options: StateProvider(device)).state
    provider = view._resolved_provider
    assert isinstance(provider, StateProvider)
    assert isinstance(provider.capabilities, set)
    assert type(provider.requires_coordinate_tools) is bool
    assert type(provider.requires_active_contract_for_coords) is bool


def test_compiled_action_carries_a_pydantic_argument_schema():
    # agent/tools/registry.py:_route, ToolRegistry.validate
    async def action(index: int) -> None:
        return None

    schema = Action.compile("device", "test_action", action, Declaration())._arguments
    assert issubclass(schema, BaseModel)
    assert schema.model_validate({"index": 1}).model_dump() == {"index": 1}


def test_adapter_accepts_single_tool_request_options():
    # agent/action_selector/llm/calls.py:achat
    options = {"chat_history": [], "allow_parallel_tool_calls": False, "tool_required": True}
    prepare = signature(FunctionCallingLLM._prepare_chat_with_tools_compat)
    assert options.keys() <= prepare.parameters.keys()
    prepare.bind(object(), [], **options)


def test_memory_estimator_accepts_message_input():
    # agent/memory/memory.py:_RunMemory._estimate_token_count; agent/logs/memory.py:report_memory
    signature(Memory._estimate_token_count).bind(object(), [])


@pytest.mark.parametrize("adapter", [OpenAIResponses, OpenAILike, OpenAILikeResponses, OpenRouter])
def test_openai_adapter_accepts_async_credential_selector(adapter):
    # agent/action_selector/llm/models.py:_Responses, _Chat, _CompatibleResponses, _Router
    signature(adapter._get_credential_kwargs).bind(object(), True)

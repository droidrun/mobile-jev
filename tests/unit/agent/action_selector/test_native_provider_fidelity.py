import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM

from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.agent.action_selector.llm.factory import build_llm
from mobilerun_agents.agent.action_selector.llm.models import MODELS
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import fake_tools


def configured_selector(monkeypatch, provider, model):
    monkeypatch.setenv(MODELS[provider].env_var, "fake")
    config = MobileConfig()
    config.profiles["default"] = LLMProfile(provider=provider, model=model)
    with ConfigLoader.override(config):
        return asyncio.run(default_setup()).selector


@pytest.mark.parametrize(
    "provider,model,api",
    [
        ("deepseek", "deepseek-flash", "chat_completions"),
        ("anthropic", "claude-sonnet-4-6", None),
        ("openai", "gpt-4o", "responses"),
        ("openai_like", "local", None),
        ("gemini", "gemini-3.8-flash", None),
    ],
)
def test_native_provider_declarations(monkeypatch, provider, model, api):
    selector = configured_selector(monkeypatch, provider, model)
    assert selector.supports_native_tool_calls is (api is not None)
    if api is not None:
        assert selector.native_tool_api == api


def test_undeclared_external_llm_is_not_guessed():
    class AdvertisesGenericFunctions(MockLLM):
        @property
        def metadata(self):
            return super().metadata.model_copy(update={"is_function_calling_model": True})

    assert not LlamaIndexLLMSelector(AdvertisesGenericFunctions()).supports_native_tool_calls


@pytest.mark.parametrize("stream", [False, True])
def test_direct_gpt6_selector_sends_native_tools_and_keeps_function_call(stream):
    requests = []
    output = [
        {"type": "reasoning", "id": "rs_unit", "summary": []},
        {
            "type": "message",
            "id": "msg_unit",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Wait for the screen.", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_unit",
            "call_id": "call_unit",
            "name": "wait",
            "arguments": "{}",
            "status": "completed",
        },
    ]
    response = {
        "id": "resp_unit",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-6-luna",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "tools": [],
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 12,
            "output_tokens_details": {"reasoning_tokens": 4},
            "total_tokens": 32,
        },
    }

    def send(request):
        requests.append((request.url.path, json.loads(request.content)))
        if stream:
            event = {"type": "response.completed", "sequence_number": 1, "response": response}
            return httpx.Response(
                200,
                text=f"data: {json.dumps(event)}\n\n",
                headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=response)

    async def select():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            llm = await build_llm("openai", "gpt-6-luna", api_key="fake", async_http_client=client)
            turn = Turn(goal="Wait", tools=fake_tools("wait"))
            return await turn.select(LlamaIndexLLMSelector(llm, vision="on"), stream=stream)

    selection = asyncio.run(select())
    [(path, body)] = requests
    assert path == "/v1/responses"
    assert body.get("tool_choice") == "required", {
        "tool_choice": body.get("tool_choice"),
        "tool_count": len(body.get("tools") or []),
        "selected_actions": len(selection.calls),
        "xml_prompt": "<function_calls>" in json.dumps(body["input"]),
    }
    assert [(tool["type"], tool["name"]) for tool in body["tools"]] == [("function", "wait")]
    assert body["stream"] is stream and body["parallel_tool_calls"] is True
    assert "<function_calls>" not in json.dumps(body["input"])
    assert selection.thought == "Wait for the screen."
    assert [(action.tool_name, action.tool_id) for action in selection.calls] == [
        ("wait", "call_unit")
    ]


@pytest.mark.parametrize("provider", ["openai_like", "ollama", "gemini"])
def test_profile_base_url_reaches_provider_options(monkeypatch, provider):
    from mobilerun_agents.agent.action_selector import loader

    url = "https://unit.invalid/v1"
    config = MobileConfig()
    config.profiles["default"] = LLMProfile(
        provider=provider,
        model="test",
        base_url=url,
    )
    build = AsyncMock(return_value=MockLLM())
    monkeypatch.setattr(loader, "build_llm", build)
    with ConfigLoader.override(config):
        asyncio.run(default_setup())
    options = build.await_args.kwargs
    if provider == "gemini":
        assert options["http_options"].base_url == url
    else:
        assert options["api_base" if provider == "openai_like" else "base_url"] == url


@pytest.mark.parametrize(
    "provider,model,choice",
    [("deepseek", "deepseek-flash", "auto"), ("openai", "gpt-4o", "required")],
)
def test_declared_native_api_controls_wire_shape(provider, model, choice, monkeypatch):
    selector = configured_selector(monkeypatch, provider, model)
    reply = ChatResponse(
        message=ChatMessage(
            role="assistant", blocks=[ToolCallBlock(tool_name="wait", tool_kwargs={})]
        ),
        additional_kwargs={"usage": {"input_tokens": 10}},
    )
    transport = AsyncMock(return_value=reply)
    monkeypatch.setattr(type(selector.llm), "achat_with_tools", transport)
    actions = fake_tools("wait")
    result = asyncio.run(Turn(goal="Wait", tools=actions).select(selector))
    assert result.calls[0].tool_name == "wait"
    transport.assert_awaited_once()
    kwargs = transport.await_args.kwargs
    assert tuple(transport.await_args.args[0]) == tuple(actions)
    assert actions[0].metadata.get_name() == "wait"
    assert kwargs["tool_required"] == (choice == "required")


@pytest.mark.parametrize(
    "provider,model,api",
    [
        ("openai", "gpt-4o", "responses"),
        ("deepseek", "deepseek-flash", "chat_completions"),
        ("anthropic", "claude-sonnet-4-6", "anthropic"),
        ("gemini", "gemini-3.8-flash", "gemini"),
    ],
)
@pytest.mark.parametrize(
    "arguments,valid",
    [
        ({"index": 3}, True),
        ({"index": 3.0}, True),
        ({"index": 3.5}, False),
        ('{"index":3', False),
        ({}, False),
        ({"index": 3, "other": 0}, False),
        ({"index": "3"}, False),
    ],
)
def test_native_adapters_preserve_decoded_arguments_and_reject_invalid_calls(
    provider, model, api, arguments, valid, monkeypatch
):
    from mobilerun_agents import tool
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.config.user_config import ToolsConfig

    @tool
    async def choose(index: int) -> int:
        """Choose one index."""
        return index

    turn = Turn(goal="Choose")
    turn.registry = build_registry(turn.device, [choose], ToolsConfig())
    llm = configured_selector(monkeypatch, provider, model).llm
    selector = LlamaIndexLLMSelector(llm, native_tool_api=api)
    message = ChatMessage(
        role="assistant",
        blocks=[ToolCallBlock(tool_name="choose", tool_call_id="c", tool_kwargs=arguments)],
    )
    transport = AsyncMock(return_value=ChatResponse(message=message))
    monkeypatch.setattr(type(llm), "achat", transport)
    if valid:
        result = asyncio.run(turn.select(selector))
        assert result.calls[0].tool_kwargs == {"index": 3}
        assert result.calls[0].tool_id == "c"
    else:
        with pytest.raises(InvalidResponse):
            asyncio.run(turn.select(selector))
    transport.assert_awaited_once()


def test_single_tool_policy_does_not_silently_discard_second_native_call(monkeypatch):
    selector = configured_selector(monkeypatch, "deepseek", "deepseek-flash")
    reply = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ToolCallBlock(tool_name="wait", tool_kwargs={}) for _ in range(2)],
        )
    )
    monkeypatch.setattr(type(selector.llm), "achat", AsyncMock(return_value=reply))
    with pytest.raises(InvalidResponse):
        asyncio.run(
            Turn(goal="Wait", tools=fake_tools("wait")).select(selector, parallel_tools=False)
        )

import asyncio
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM

from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.loader import build_selector
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig
from tests.unit.agent.selector_fixtures import selection_run
from tests.unit.device_fixtures import fake_tools


def configured_selector(provider, model):
    config = MobileConfig()
    options = {"api_key": "fake"}
    if provider == "gemini":
        options.update(max_tokens=2048, context_window=8192)
    config.llm_profiles[config.agent.selector] = LLMProfile(
        provider=provider, model=model, kwargs=options
    )
    return asyncio.run(build_selector(config))


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
def test_native_provider_declarations(provider, model, api):
    selector = configured_selector(provider, model)
    assert selector.supports_native_tool_calls is (api is not None)
    if api is not None:
        assert selector.native_tool_api == api


def test_undeclared_external_llm_is_not_guessed():
    class AdvertisesGenericFunctions(MockLLM):
        @property
        def metadata(self):
            return super().metadata.model_copy(update={"is_function_calling_model": True})

    assert not LlamaIndexLLMSelector(AdvertisesGenericFunctions()).supports_native_tool_calls


@pytest.mark.parametrize("provider", ["openai_like", "ollama", "gemini"])
def test_profile_base_url_reaches_provider_options(monkeypatch, provider):
    from mobilerun_agents.agent.action_selector import loader

    url = "https://unit.invalid/v1"
    config = MobileConfig()
    config.llm_profiles["default"] = LLMProfile(
        provider=provider,
        model="test",
        base_url=url,
        kwargs={"http_options": {"timeout": 123}} if provider == "gemini" else {},
    )
    build = AsyncMock(return_value=MockLLM())
    monkeypatch.setattr(loader, "build_llm", build)
    asyncio.run(build_selector(config))
    options = build.await_args.kwargs
    if provider == "gemini":
        assert (options["http_options"].base_url, options["http_options"].timeout) == (url, 123)
    else:
        assert options["api_base" if provider == "openai_like" else "base_url"] == url


@pytest.mark.parametrize(
    "provider,model,choice",
    [("deepseek", "deepseek-flash", "auto"), ("openai", "gpt-4o", "required")],
)
def test_declared_native_api_controls_wire_shape(provider, model, choice, monkeypatch):
    selector = configured_selector(provider, model)
    reply = ChatResponse(
        message=ChatMessage(
            role="assistant", blocks=[ToolCallBlock(tool_name="wait", tool_kwargs={})]
        ),
        additional_kwargs={"usage": {"input_tokens": 10}},
    )
    transport = AsyncMock(return_value=reply)
    monkeypatch.setattr(type(selector.shared_llm), "achat_with_tools", transport)
    actions = fake_tools("wait")
    result = asyncio.run(selector.select(selection_run(goal="Wait", tools=actions)))
    assert result.response is reply
    assert result.actions[0].name == "wait"
    assert result.usage.request_tokens == 10
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
        ('{"index":3', False),
        ({}, False),
        ({"index": 3, "other": 0}, False),
        ({"index": "3"}, False),
    ],
)
def test_native_adapters_preserve_decoded_arguments_and_reject_invalid_calls(
    provider, model, api, arguments, valid, monkeypatch
):
    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.tools.registry import build_registry

    @tool
    async def choose(run: Run, index: int) -> int:
        """Choose one index."""
        return index

    run = selection_run(goal="Choose")
    run.registry = build_registry(run, [choose], None)
    llm = configured_selector(provider, model).shared_llm
    selector = LlamaIndexLLMSelector(llm, native_tool_api=api)
    message = ChatMessage(
        role="assistant",
        blocks=[ToolCallBlock(tool_name="choose", tool_call_id="c", tool_kwargs=arguments)],
    )
    transport = AsyncMock(return_value=ChatResponse(message=message))
    monkeypatch.setattr(type(llm), "achat", transport)
    result = asyncio.run(selector.select(run))
    transport.assert_awaited_once()
    assert bool(result.parse_error) is not valid
    if valid:
        assert result.actions[0].parameters == {"index": 3}
        assert result.actions[0].call_id == "c"


def test_single_tool_policy_does_not_silently_discard_second_native_call(monkeypatch):
    selector = configured_selector("deepseek", "deepseek-flash")
    reply = ChatResponse(
        message=ChatMessage(
            role="assistant",
            blocks=[ToolCallBlock(tool_name="wait", tool_kwargs={}) for _ in range(2)],
        )
    )
    monkeypatch.setattr(type(selector.shared_llm), "achat", AsyncMock(return_value=reply))
    result = asyncio.run(
        selector.select(selection_run(goal="Wait", tools=fake_tools("wait"), parallel_tools=False))
    )
    assert len(result.actions) == 2  # the workflow rejects this entire selection before dispatch

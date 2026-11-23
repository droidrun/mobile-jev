"""Exercise sampling payloads at the actual Anthropic SDK HTTP boundary."""

import asyncio
import json
from types import SimpleNamespace

import anthropic
import pytest

try:
    from anthropic._base_client import httpx2 as httpx
except ImportError:
    import httpx
from llama_index.core.llms import ChatMessage

from mobilerun.agents.llm.llm_picker import load_llm


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "model,extra_body,expected",
    [
        (
            "claude-haiku-4-5",
            {"temperature": 0.7, "metadata": {"user_id": "test"}},
            {"temperature": 0.7, "top_p": 0.6, "top_k": 10},
        ),
        ("claude-sonnet-4-6", {}, {"temperature": 0.2, "top_p": 0.6, "top_k": 10}),
        ("claude-opus-4-8", {"temperature": 0.7, "top_p": 0.3, "top_k": 1}, {}),
        ("claude-sonnet-4-6", {"model": "claude-opus-4-8", "temperature": 0.7}, {}),
    ],
)
def test_sdk_request_preserves_supported_sampling_only(async_call, model, extra_body, expected):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "READY"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    llm = load_llm(
        "Anthropic",
        model=model,
        api_key="test-key",
        temperature=0.2,
        additional_kwargs={"top_p": 0.6},
    )
    body_before = dict(extra_body)
    messages = [ChatMessage(role="user", content="Reply READY")]
    if async_call:

        async def run():
            async with anthropic.AsyncAnthropic(
                api_key="test-key",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as client:
                llm._aclient = client
                return await llm.achat(messages, top_k=10, extra_body=extra_body)

        result = asyncio.run(run())
    else:
        with anthropic.Anthropic(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            llm._client = client
            result = llm.chat(messages, top_k=10, extra_body=extra_body)

    assert result.message.content == "READY"
    payload = captured[0]
    assert {k: payload[k] for k in ("temperature", "top_p", "top_k") if k in payload} == expected
    assert payload["model"] == extra_body.get("model", model)
    assert "extra_body" not in payload
    if "metadata" in extra_body:
        assert payload["metadata"] == extra_body["metadata"]
    assert extra_body == body_before


@pytest.mark.parametrize("named_sampling", [False, True])
def test_sampling_adapts_to_installed_sdk_signature(named_sampling):
    def old_create(*, temperature=None, top_p=None, top_k=None, extra_body=None):
        pass

    def new_create(*, extra_body=None):
        pass

    llm = load_llm("Anthropic", model="claude-haiku-4-5", api_key="test-key", temperature=0.2)
    llm._client = SimpleNamespace(
        messages=SimpleNamespace(create=old_create if named_sampling else new_create)
    )
    kwargs = llm._get_all_kwargs(top_p=0.6, top_k=10, extra_body={"temperature": 0.7})
    assert ("temperature" in kwargs) is named_sampling
    assert {**kwargs, **kwargs["extra_body"]}["temperature"] == 0.7
    assert {**kwargs, **kwargs["extra_body"]}["top_p"] == 0.6
    assert {**kwargs, **kwargs["extra_body"]}["top_k"] == 10


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "tool_choice,expected",
    [
        (None, {"disable_parallel_tool_use": True, "type": "any"}),
        ("required", {"type": "any"}),
        ("auto", {"type": "auto"}),
        ({"type": "tool", "name": "Result"}, {"type": "tool", "name": "Result"}),
    ],
)
def test_structured_extraction_sends_anthropic_tool_choice(async_call, tool_choice, expected):
    from llama_index.core.program.function_program import FunctionCallingProgram
    from llama_index.core.prompts import PromptTemplate
    from pydantic import BaseModel

    class Result(BaseModel):
        value: int

    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_test",
                        "name": "Result",
                        "input": {"value": 19},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    llm = load_llm("Anthropic", model="claude-sonnet-4-6", api_key="test-key")
    prompt = PromptTemplate("Return 8 plus 11 as value.")
    program = FunctionCallingProgram.from_defaults(
        output_cls=Result, llm=llm, prompt=prompt, tool_choice=tool_choice
    )
    if async_call:

        async def run():
            async with anthropic.AsyncAnthropic(
                api_key="test-key",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as client:
                llm._aclient = client
                return (
                    await llm.astructured_predict(Result, prompt)
                    if tool_choice is None
                    else await program.acall()
                )

        result = asyncio.run(run())
    else:
        with anthropic.Anthropic(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client:
            llm._client = client
            result = llm.structured_predict(Result, prompt) if tool_choice is None else program()
    assert result.value == 19
    assert captured[0]["tool_choice"] == expected
    assert captured[0]["tools"][0]["input_schema"]["properties"]["value"]["type"] == "integer"

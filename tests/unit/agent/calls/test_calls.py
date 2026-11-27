import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, TextBlock, ToolCallBlock
from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails

from mobilerun_agents.agent.action_selector.llm.calls import (
    InvalidResponse,
    achat,
    usage_from_response,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


def reply(text=None, **kwargs):
    return ChatResponse(message=ChatMessage(role="assistant", content=text), **kwargs)


@pytest.mark.parametrize(
    "raw,extra,expected",
    [
        (
            SimpleNamespace(
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=5,
                        output_tokens=2,
                        total_tokens=11,
                        output_tokens_details={"reasoning_tokens": 4},
                    )
                )
            ),
            {},
            (5, 6, 11),
        ),
        (
            {
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 6,
                    "total_tokens": 11,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                }
            },
            {},
            (5, 6, 11),
        ),
        (
            {
                "response": {
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 2,
                        "thoughtsTokenCount": 4,
                    }
                }
            },
            {},
            (5, 6, 11),
        ),
        (
            {
                "usage_metadata": {
                    "prompt_token_count": 5,
                    "candidates_token_count": 2,
                    "thoughts_token_count": 4,
                    "total_token_count": 14,
                }
            },
            {"usage": {"completion_tokens": 2}},
            (5, 6, 14),
        ),
        ({"usage": {"input_tokens": 4, "output_tokens": 8}}, {}, (4, 8, 12)),
        (
            {
                "usage": CompletionUsage(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                    completion_tokens_details=CompletionTokensDetails(),
                )
            },
            {},
            (3, 2, 5),
        ),
        ({"prompt_eval_count": 4, "eval_count": 3}, {}, (4, 3, 7)),
        (None, {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}, (0, 0, 0)),
        (None, {}, (None, None, None)),
    ],
)
def test_usage(raw, extra, expected):
    usage = usage_from_response(reply("hi", raw=raw, additional_kwargs=extra))
    assert (usage.request_tokens, usage.response_tokens, usage.total_tokens) == expected
    assert usage.requests == 1


@pytest.mark.parametrize("stage", ["chat", "open_stream", "iterate_stream"])
def test_deadline_cancels_stalled_call(stage):
    llm = AsyncMock()

    async def stall(*args):
        await asyncio.Event().wait()

    async def chunks():
        yield reply("first")
        await stall()

    llm.achat.side_effect = llm.astream_chat.side_effect = stall
    if stage == "iterate_stream":
        llm.astream_chat.side_effect = None
        llm.astream_chat.return_value = chunks()
    with pytest.raises(TimeoutError):
        asyncio.run(achat(llm, [], timeout=0.01, stream=stage != "chat"))


def test_textless_tool_requires_offered_tools():
    llm = AsyncMock()
    response = reply()
    response.message.blocks = [ToolCallBlock(tool_name="click")]
    llm.achat.return_value = response
    with pytest.raises(InvalidResponse):
        asyncio.run(achat(llm, []))
    assert asyncio.run(achat(llm, [], tools=[{"name": "click"}])) is response


def test_stream_preserves_delta_text_and_final_usage():
    async def chunks():
        yield reply("he", delta="he")
        yield reply("llo", delta="llo")
        yield reply(raw={"response": {"usage": {"output_tokens": 4}}})

    llm = AsyncMock()
    llm.astream_chat.return_value = chunks()
    response = asyncio.run(achat(llm, [], stream=True))
    assert response.message.content == "hello"
    assert usage_from_response(response).response_tokens == 4


def test_stream_preserves_cumulative_block_order():
    response = reply(delta="hello")
    response.message.blocks = [TextBlock(text="hello"), ToolCallBlock(tool_name="click")]

    async def chunks():
        yield response

    llm = AsyncMock()
    llm.astream_chat.return_value = chunks()
    assert asyncio.run(achat(llm, [], stream=True)).message == response.message

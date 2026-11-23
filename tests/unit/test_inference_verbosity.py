import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.prompts import PromptTemplate
from mobilerun.logs.handlers import VERBOSE
from pydantic import BaseModel

from mobilerun.agents.llm.inference import (
    acall_with_retries,
    acomplete_with_retries,
    astructured_predict_with_retries,
)


class _Result(BaseModel):
    value: str


class _LLM:
    async def achat(self, **kwargs: object) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="assistant", content="chat reply"))

    async def astream_chat(self, **kwargs: object) -> AsyncIterator[ChatResponse]:
        async def chunks() -> AsyncIterator[ChatResponse]:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content="chat reply"),
                delta="chat reply",
            )

        return chunks()

    async def acomplete(self, prompt: str) -> CompletionResponse:
        return CompletionResponse(text="completion reply")

    async def astream_complete(self, prompt: str) -> AsyncIterator[CompletionResponse]:
        async def chunks() -> AsyncIterator[CompletionResponse]:
            yield CompletionResponse(text="completion reply", delta="completion reply")

        return chunks()

    async def astructured_predict(
        self, output_cls: type[_Result], prompt: PromptTemplate, **kwargs: object
    ) -> _Result:
        return output_cls(value="structured reply")


@pytest.mark.parametrize(
    "kind,stream",
    [
        ("chat", False),
        ("chat", True),
        ("completion", False),
        ("completion", True),
        ("structured", False),
    ],
)
def test_inference_diagnostics_are_verbose_and_stream_flags_survive(
    kind: str, stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    logger = logging.getLogger("mobilerun")
    caplog.set_level(VERBOSE, logger="mobilerun")
    logger.addHandler(caplog.handler)
    try:
        llm = _LLM()
        if kind == "chat":
            asyncio.run(acall_with_retries(llm, [], stream=stream))
        elif kind == "completion":
            asyncio.run(acomplete_with_retries(llm, "prompt", stream=stream))
        else:
            asyncio.run(astructured_predict_with_retries(llm, _Result, PromptTemplate("test")))
        records = [record for record in caplog.records if record.name == "mobilerun"]
        assert records
        assert all(record.levelno == VERBOSE for record in records)
        if stream:
            assert any(record.__dict__.get("stream") is True for record in records)
            assert any(record.__dict__.get("stream_end") is True for record in records)
    finally:
        logger.removeHandler(caplog.handler)

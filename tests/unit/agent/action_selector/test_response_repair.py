import asyncio
from unittest.mock import AsyncMock

from llama_index.core.llms import ChatMessage, ChatResponse

from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from tests.unit.agent.selector_fixtures import selection_run


def test_invalid_reply_is_returned_intact_for_workflow_repair():
    reply = ChatResponse(
        message=ChatMessage(role="assistant", content='<function_calls><invoke name="wait">'),
        additional_kwargs={"usage": {"input_tokens": 10}},
    )

    class Chat(LLMSelector):
        _achat = AsyncMock(return_value=reply)

    from mobilerun_agents import Run, tool
    from mobilerun_agents.agent.tools.registry import build_registry
    from tests.unit.device_fixtures import tool_run

    @tool(sensitive_parameters=frozenset({"text"}))
    async def private_text(run: Run, text: str) -> None:
        """Use a secret."""

    run = tool_run()
    private = build_registry(run, [private_text], None).by_name["private_text"]
    chat = Chat()
    result = asyncio.run(chat.select(selection_run(goal="Wait", tools=[private])))
    chat._achat.assert_awaited_once()
    assert chat._achat.await_args.kwargs["log_response"] is False
    assert result.response is reply
    assert result.parse_error and not result.actions

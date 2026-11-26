import asyncio
from unittest.mock import AsyncMock

import pytest
from llama_index.core.llms import ChatMessage, ChatResponse

from mobilerun_agents.agent.action_selector.llm.base import LLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from tests.unit.agent.action_selector.fixtures import Turn


def test_invalid_reply_returns_safe_signal_for_workflow_repair():
    reply = ChatResponse(
        message=ChatMessage(role="assistant", content='<function_calls><invoke name="wait">'),
        additional_kwargs={"usage": {"input_tokens": 10}},
    )

    class Chat(LLMSelector):
        _achat = AsyncMock(return_value=reply)

    from mobilerun_agents import tool
    from mobilerun_agents.agent.tools.registry import build_registry
    from mobilerun_agents.config.user_config import ToolsConfig

    @tool(sensitive_parameters=frozenset({"text"}))
    async def private_text(text: str) -> None:
        """Use a secret."""

    turn = Turn(goal="Wait")
    turn.registry = build_registry(turn.device, [private_text], ToolsConfig())
    chat = Chat()
    with pytest.raises(InvalidResponse) as error:
        asyncio.run(turn.select(chat))
    chat._achat.assert_awaited_once()
    assert chat._achat.await_args.kwargs["log_response"] is False
    assert "<invoke" not in str(error.value)

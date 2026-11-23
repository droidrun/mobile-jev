from llama_index.core.base.llms.types import ChatMessage, TextBlock, ToolCallBlock

from mobilerun.agents.trajectory.writer import make_serializable


def test_message_serialization_reads_text_blocks():
    message = ChatMessage(role="assistant", blocks=[TextBlock(text="hello")])
    assert make_serializable(message) == {"role": "assistant", "content": "hello"}


def test_unrelated_class_named_chat_message_is_not_treated_as_llama_message():
    OtherMessage = type("ChatMessage", (), {"content": "not an llm message"})
    assert make_serializable(OtherMessage()) == {}


def test_message_serialization_ignores_non_text_blocks():
    message = ChatMessage(role="assistant", blocks=[ToolCallBlock(tool_name="click")])
    assert make_serializable(message) == {"role": "assistant", "content": ""}

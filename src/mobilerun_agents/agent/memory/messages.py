import json

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
from llama_index.core.memory import Memory
from llama_index.core.tools import ToolSelection

from mobilerun_agents.agent.workflows.core.state import ActionRecord


async def remember(memory: Memory, call: ToolSelection, record: ActionRecord) -> None:
    """Append one complete tool turn so the FIFO never splits its call and result."""
    call_id, name, args = call.tool_id, call.tool_name, call.tool_kwargs
    index = args.get("index")
    target = f" [{index}]" if type(index) is int else ""
    status = "✓" if record.ok else "✗"
    result = f"{record.step} · {name}{target} {status}\n{record.summary}"
    if record.thought:
        result += f"\nThought: {record.thought}"
    await memory.aput_messages(
        [
            # LlamaIndex's eviction keeps complete turns starting at a user message.
            # This marker carries no screen content.
            ChatMessage(role="user", content="Action taken:"),
            ChatMessage(
                role="assistant",
                blocks=[ToolCallBlock(tool_call_id=call_id, tool_name=name, tool_kwargs=args)],
            ),
            ChatMessage(role="tool", content=result, additional_kwargs={"tool_call_id": call_id}),
        ]
    )


def text_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Project native calls into readable text for models not consuming tool roles."""
    projected = []
    for message in messages:
        calls: list[str] = []
        for block in message.blocks:
            match block:
                case ToolCallBlock(tool_name=name, tool_kwargs=kwargs):
                    calls.append(f"{name}({json.dumps(kwargs, ensure_ascii=False)})")
        if message.role == "tool":
            message = ChatMessage(role="assistant", content=message.content)
        elif calls:
            message = ChatMessage(
                role=message.role, content="\n".join(filter(None, [message.content, *calls]))
            )
        projected.append(message)
    return projected

import json
from uuid import uuid4

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock

from mobilerun_agents.agent.workflows.core.state import ActionOutcome, ActionRecord


def action_messages(record: ActionOutcome, wire_name: str) -> list[ChatMessage]:
    # Parallel actions can share a workflow step number, but not a tool-call ID.
    match record:
        case ActionRecord(call_id=call_id):
            pass
        case _:
            call_id = f"call_{uuid4().hex}"
    index = record.args.get("index")
    target = f" [{index}]" if type(index) is int else ""
    change = {True: "yes", False: "no", None: "unknown"}[record.screen_changed]
    status = "✓" if record.ok else "✗"
    unchanged = " · no change" if record.screen_changed is False else ""
    result = f"{record.step} · {wire_name}{target} {status}{unchanged}\n{record.summary}\nscreen changed: {change}"
    if record.thought:
        result += f"\nThought: {record.thought}"
    return [
        # LlamaIndex's eviction keeps complete turns starting at a user message.
        # This marker carries no screen content.
        ChatMessage(role="user", content="Action taken:"),
        ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(tool_call_id=call_id, tool_name=wire_name, tool_kwargs=record.args)
            ],
        ),
        ChatMessage(role="tool", content=result, additional_kwargs={"tool_call_id": call_id}),
    ]


def text_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Project native calls into readable text for models not consuming tool roles."""
    projected = []
    for message in messages:
        calls = [
            f"{block.tool_name}({json.dumps(block.tool_kwargs, ensure_ascii=False)})"
            for block in message.blocks
            if isinstance(block, ToolCallBlock)
        ]
        if message.role == "tool":
            message = ChatMessage(role="assistant", content=message.content)
        elif calls:
            message = ChatMessage(
                role=message.role, content="\n".join(filter(None, [message.content, *calls]))
            )
        projected.append(message)
    return projected

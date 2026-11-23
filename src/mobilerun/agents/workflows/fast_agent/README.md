# agents/workflows/fast_agent/

Runs a tool-calling agent and parses its tool calls.

Keep fast-agent workflow and tool-call parsing here; shared actions and inference helpers live in `tools/` and `llm/`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports `FastAgent`. |
| `events.py` | `FastAgentInputEvent`, `FastAgentResponseEvent`, `FastAgentToolCallEvent`, `FastAgentOutputEvent`, `FastAgentEndEvent` — events for the fastagent workflow. |
| `fast_agent.py` | `FastAgent` — fastagent — tool-calling agent for device interaction. |
| `native_tools.py` | `native_tool_kwargs`, `parse_native_tool_calls`, `tool_result_messages` — native function-calling support for fastagent. |
| `xml_parser.py` | `ToolCall`, `ToolCallParseStatus`, `ToolCallParseResult`, `ToolResult`, `parse_tool_calls`, `parse_tool_calls_detailed`, `format_tool_results`, `format_tool_calls`, `extract_add_memory` — xml tool-call parsing and result formatting. |


See [agent overview](../README.md).

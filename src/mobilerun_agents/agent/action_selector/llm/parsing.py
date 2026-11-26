"""Parse XML/native calls without interpreting arguments until the action is known."""

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
from mobilerun.models.context import JSONValue
from pydantic import FiniteFloat, TypeAdapter, ValidationError

from mobilerun_agents.agent.action_selector.llm.calls import InvalidResponse
from mobilerun_agents.config.prompts import render_selector_prompt


class ToolCallParseStatus(str, Enum):
    """Classification of tool-call markup in an LLM response."""

    VALID = "valid"
    NO_MARKUP = "no_markup"
    MALFORMED = "malformed"


logger = logging.getLogger("mobilerun")

OPEN_TAG = "<function_calls>"
CLOSE_TAG = "</function_calls>"

_PARAM_RE = re.compile(
    r"""(<parameter\s+name=(?:"[^"]*"|'[^']*')\s*>)(.*?)(</parameter>)""",
    re.DOTALL,
)
_PARAM_OPEN_RE = re.compile(r"<parameter(?=[\s>/])")

_DSML_MARKUP_PATTERN = r"(?:<|＜)\s*[/／]?\s*｜+DSML｜+"
_TOOL_CALL_MARKUP_RE = re.compile(
    r"(?:<|＜)\s*[/／]?\s*(?:function_calls?|tool_calls?|invoke|parameter)\b"
    r"|" + _DSML_MARKUP_PATTERN,
    re.IGNORECASE,
)
_CANONICAL_BLOCK_TAGS = ("<invoke", "</invoke", "<parameter", "</parameter")


@dataclass
class ToolCall:
    """A parsed tool invocation from the LLM response."""

    name: str
    parameters: dict[str, JSONValue] = field(default_factory=dict)
    error: Optional[str] = None
    # Native tool calls carry the provider's id for matching results.
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolCallParseResult:
    """Distinguish missing and invalid markup from a valid tool-call block."""

    thought: str
    calls: List[ToolCall]
    status: ToolCallParseStatus


def parse_tool_calls_detailed(text: str) -> ToolCallParseResult:
    """Parse tool calls and classify whether markup was absent, valid, or malformed.

    Only the first tool-call block is parsed: the first <function_calls>
    always starts it. If it is malformed (including any tool-call markup
    before that opener), the result is malformed even when a later block is
    valid.
    """
    opener = text.find(OPEN_TAG)
    before = text if opener == -1 else text[:opener]
    leading = _TOOL_CALL_MARKUP_RE.search(before)
    if leading:
        return ToolCallParseResult(
            thought=before[: leading.start()].strip(),
            calls=[],
            status=ToolCallParseStatus.MALFORMED,
        )

    thought = before.strip()
    if opener == -1:
        return ToolCallParseResult(thought, [], ToolCallParseStatus.NO_MARKUP)
    start = opener + len(OPEN_TAG)
    close = text.find(CLOSE_TAG, start)
    if close == -1 or text.find(OPEN_TAG, start, close) != -1:
        return ToolCallParseResult(thought, [], ToolCallParseStatus.MALFORMED)

    block = text[start:close].strip()
    # Parameter-value sanitization intentionally preserves raw text such as
    # code and literal DSML. Provider-specific markup is malformed only
    # when it occurs outside a trustworthy canonical parameter payload.
    calls = _parse_tool_call_block(block) if block and not _has_noncanonical_markup(block) else []
    return ToolCallParseResult(
        thought=thought,
        calls=calls,
        status=ToolCallParseStatus.VALID if calls else ToolCallParseStatus.MALFORMED,
    )


_ADD_MEMORY_RE = re.compile(
    r"<add_memory(?:\s+[^>]*)?>(.*?)</add_memory>",
    re.DOTALL,
)


def extract_add_memory(text: str) -> str:
    """Extract and combine content from all ``<add_memory>`` tags in LLM response text."""
    matches = _ADD_MEMORY_RE.findall(text)
    if not matches:
        return ""
    return "\n".join(m.strip() for m in matches if m.strip())


def _parse_tool_call_block(block: str) -> List[ToolCall]:
    block = _sanitize_param_content(block)
    # Comments and processing instructions are dropped by the XML parser,
    # which would hide them from the checks below.
    if "<!" in block or "<?" in block:
        logger.warning("Comment or declaration in tool call XML block, skipping block")
        return []

    try:
        root = ET.fromstring(f"<root>{block}</root>")
    except (ET.ParseError, UnicodeError):
        logger.warning("Failed to parse tool call XML block, skipping")
        return []

    if _has_text_outside_parameters(root):
        logger.warning("Text outside tool calls in XML block, skipping block")
        return []

    for child in root:
        if child.tag != "invoke":
            logger.warning(
                "Unexpected <%s> element in tool call XML block, skipping block",
                child.tag,
            )
            return []

    calls: List[ToolCall] = []
    for invoke in root.findall("invoke"):
        name = invoke.get("name", "")
        if not name:
            logger.warning("Tool call without a name in XML block, skipping block")
            return []

        params: dict[str, JSONValue] = {}
        for param in invoke:
            param_name = param.get("name", "")
            if param.tag != "parameter" or not param_name or len(param):
                logger.warning("Invalid tool call parameter in XML block, skipping block")
                return []
            params[param_name] = param.text or ""

        calls.append(ToolCall(name=name, parameters=params))
    return calls


def _has_text_outside_parameters(root: ET.Element) -> bool:
    """Return whether the block has text anywhere except inside parameter values."""
    texts = [root.text]
    for invoke in root:
        texts += [invoke.tail, invoke.text]
        texts += [param.tail for param in invoke]
    return any(text and text.strip() for text in texts)


def _has_noncanonical_markup(block: str) -> bool:
    """Return whether tool-call markup other than plain <invoke>/<parameter>
    tags (DSML, full-width or alternate tags) occurs outside a trustworthy
    parameter payload."""
    payload_spans: List[Tuple[int, int]] = []
    for parameter in _PARAM_RE.finditer(block):
        payload = parameter.group(2)
        # A nested parameter opener means the regex may have crossed
        # from a missing close tag into a later parameter. Do not hide DSML in
        # that span from structural validation.
        if _PARAM_OPEN_RE.search(payload):
            continue
        payload_spans.append(parameter.span(2))

    for marker in _TOOL_CALL_MARKUP_RE.finditer(block):
        if marker.group(0) in _CANONICAL_BLOCK_TAGS:
            continue
        if not any(start <= marker.start() and marker.end() <= end for start, end in payload_spans):
            return True
    return False


def _sanitize_param_content(block: str) -> str:
    """Escape XML-unsafe characters inside parameter values.

    Parameter values often contain raw code or text with <, >, &
    which would break XML parsing. This escapes content inside
    <parameter> tags only, leaving the XML structure intact.
    """

    def _escape(m: re.Match[str]) -> str:
        pre, content, post = m.group(1), m.group(2), m.group(3)
        clean = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return pre + clean + post

    return _PARAM_RE.sub(_escape, block)


_JSON = TypeAdapter(JSONValue)
_ARGUMENTS = TypeAdapter(dict[str, JSONValue])
_TYPES: dict[str, TypeAdapter[Any]] = {
    "boolean": TypeAdapter(bool),
    "integer": TypeAdapter(int),
    "number": TypeAdapter(FiniteFloat),
    "string": TypeAdapter(str),
    "array": TypeAdapter(list[JSONValue]),
    "object": _ARGUMENTS,
    "null": TypeAdapter(None),
}


def _argument(schema: Mapping[str, Any], value: JSONValue, defs: Mapping[str, Any]) -> JSONValue:
    if "$ref" in schema:
        schema = defs[schema["$ref"].removeprefix("#/$defs/")]
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                return _argument(branch, value, defs)
            except ValueError:
                continue
        raise ValueError
    if kind := schema.get("type"):
        if kind == "integer" and type(value) is float and value.is_integer():
            value = int(value)
        value = _TYPES[kind].validate_python(value, strict=True)
    if "enum" in schema or "const" in schema:
        choices = schema["enum"] if "enum" in schema else [schema["const"]]
        if not any(type(value) is type(choice) and value == choice for choice in choices):
            raise ValueError
    match value:
        case list() if "items" in schema:
            return [_argument(schema["items"], item, defs) for item in value]
        case dict():
            if not set(schema.get("required", ())) <= value.keys():
                raise ValueError
            properties = schema.get("properties", {})
            extra = schema.get("additionalProperties", True)
            result = {}
            for key, item in value.items():
                child = properties.get(key, extra)
                if child is False:
                    raise ValueError
                result[key] = item if child is True else _argument(child, item, defs)
            return result
    return value


def _wire_argument(
    schema: Mapping[str, Any], value: JSONValue, defs: Mapping[str, Any]
) -> JSONValue:
    try:
        return _argument(schema, value, defs)
    except ValueError:
        if type(value) is not str:
            raise
    # Preserve strings when the schema allows them. Otherwise decode exactly one JSON
    # literal (never turn an invalid boolean into False, or a scalar into a list).
    decoded = _JSON.validate_json(value)
    if type(decoded) is str:
        raise ValueError
    return _argument(schema, decoded, defs)


def coerce_arguments(
    schema: Mapping[str, Any], raw: Mapping[str, JSONValue], *, textual: bool = True
) -> dict[str, JSONValue]:
    """Decode wire values against this action's advertised argument schema only."""
    properties = schema.get("properties", {})
    try:
        if not set(schema.get("required", ())) <= raw.keys():
            raise ValueError
        normalize = _wire_argument if textual else _argument
        return {
            key: normalize(properties[key], value, schema.get("$defs", {}))
            for key, value in raw.items()
        }
    except (ValueError, KeyError):
        raise InvalidResponse(
            render_selector_prompt("assets/prompts/llm/invalid_arguments.jinja2")
        ) from None


def parse_native_tool_calls(message: ChatMessage) -> ToolCallParseResult:
    """Split a native tool-calling response into its text and tool calls."""
    calls = [_to_tool_call(block) for block in message.blocks if block.block_type == "tool_call"]
    return ToolCallParseResult(
        thought=(message.content or "").strip(),
        calls=calls,
        status=ToolCallParseStatus.VALID if calls else ToolCallParseStatus.NO_MARKUP,
    )


def _to_tool_call(block: ToolCallBlock) -> ToolCall:
    name = block.tool_name
    call_id = block.tool_call_id
    arguments = block.tool_kwargs
    try:
        parameters = (
            _ARGUMENTS.validate_json(arguments)
            if type(arguments) is str
            else _ARGUMENTS.validate_python(arguments, strict=True)
        )
    except ValidationError:
        return ToolCall(
            name=name,
            error=render_selector_prompt("assets/prompts/llm/invalid_arguments.jinja2"),
            call_id=call_id,
        )
    return ToolCall(name=name, parameters=parameters, call_id=call_id)

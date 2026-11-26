"""Persist the native LlamaIndex context without changing the live state."""

import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from workflows import Context
from workflows.context.serializers import JsonSerializer

from mobilerun_agents.agent.logs.redaction import Redactor


def _redact_serialized(value: Any, redact: Redactor) -> Any:
    match value:
        case {"qualified_name": _, "value": payload} if value.get("__is_pydantic") or value.get(
            "__is_component"
        ):
            # Type metadata and Pydantic field names must survive Context.from_dict.
            return {**value, "value": redact.value(payload, freeform=False)}
        case dict():
            return {
                redact.text(key): _redact_serialized(item, redact) for key, item in value.items()
            }
        case list():
            return [_redact_serialized(item, redact) for item in value]
        case _:
            return redact.value(value)


class _CheckpointSerializer(JsonSerializer):
    def __init__(self, redact: Redactor) -> None:
        super().__init__()
        self._redact = redact

    def serialize(self, value: Any) -> str:
        return json.dumps(_redact_serialized(self.serialize_value(value), self._redact))


def _atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(json.dumps(data, ensure_ascii=True, allow_nan=False).encode())
            handle.close()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


async def save_context(path: Path, ctx: Context[Any], redact: Redactor) -> dict[str, Any]:
    data = ctx.to_dict(serializer=_CheckpointSerializer(redact))
    write = asyncio.create_task(asyncio.to_thread(_atomic_json, path, data))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # Do not let an old write race a final, newly redacted checkpoint.
        await write
        raise
    return data


async def save_macro(path: Path, data: dict[str, Any]) -> None:
    await asyncio.to_thread(_atomic_json, path, data)

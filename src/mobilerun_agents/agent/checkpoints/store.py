"""Persist the native LlamaIndex context without changing the live state."""

import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from workflows import Context


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


async def save_context(path: Path, ctx: Context[Any]) -> dict[str, Any]:
    data = ctx.to_dict()
    write = asyncio.create_task(asyncio.to_thread(_atomic_json, path, data))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # Wait for the atomic write before propagating cancellation.
        await write
        raise
    return data


async def save_macro(path: Path, data: dict[str, Any]) -> None:
    await asyncio.to_thread(_atomic_json, path, data)

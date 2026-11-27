"""Render agent templates in a strict, immutable Jinja sandbox."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mobilerun_agents.config.paths import resolve_prompt_path

_ENVIRONMENT = ImmutableSandboxedEnvironment(
    undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False
)


@lru_cache(maxsize=None)
def _load_template(path: Path) -> Template:
    source = path.read_text(encoding="utf-8")
    return _ENVIRONMENT.from_string(source)


def render_prompt(path: str, **variables: Any) -> str:
    template = _load_template(resolve_prompt_path(path))
    return template.render(**variables).strip()

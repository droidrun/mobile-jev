"""Render agent templates in a strict, immutable Jinja sandbox."""

from typing import Any

from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mobilerun_agents.config.paths import resolve_prompt_path

_ENVIRONMENT = ImmutableSandboxedEnvironment(
    undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False
)


def render_selector_prompt(path: str, **variables: Any) -> str:
    template = resolve_prompt_path(path).read_text(encoding="utf-8")
    return _ENVIRONMENT.from_string(template).render(**variables).strip()

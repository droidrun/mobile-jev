"""Mask an individual workflow value using the core secret scope."""

from collections.abc import Callable

from mobilerun.models.context import JSONValue
from mobilerun.utils.redaction import mask_step_args


def mask_value(value: JSONValue, redact_code: Callable[[str], str] | None = None) -> JSONValue:
    def contains_code(text: str) -> bool:
        return redact_code is not None and redact_code(text) != text

    [masked] = mask_step_args({"": value}, contains_code=contains_code).values()
    return masked

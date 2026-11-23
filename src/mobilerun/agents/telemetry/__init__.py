"""Lazy access to telemetry events and tracking functions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobilerun.agents.telemetry.events import (
        MobileAgentFinalizeEvent,
        MobileAgentInitEvent,
        PackageVisitEvent,
    )
    from mobilerun.agents.telemetry.tracker import capture, flush, print_telemetry_message

__all__ = [
    "capture",
    "flush",
    "MobileAgentInitEvent",
    "MobileAgentFinalizeEvent",
    "PackageVisitEvent",
    "print_telemetry_message",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = (
        "events"
        if name in {"MobileAgentInitEvent", "MobileAgentFinalizeEvent", "PackageVisitEvent"}
        else "tracker"
    )
    return getattr(import_module(f"{__name__}.{module}"), name)

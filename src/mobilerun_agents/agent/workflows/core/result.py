"""Shared result contract for agent runs."""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


@dataclass(frozen=True)
class StepEvidence:
    """Run-local observation; the terminal observation may have no selection."""

    step: int
    package: str | None
    activity: str | None
    title: str | None = field(repr=False)
    fingerprint: tuple[str, str]
    thought: str | None = field(default=None, repr=False)


class ScreenVisit(BaseModel):
    package: str | None
    activity: str | None
    title: str | None
    count: int = 1


class FailureSummary(BaseModel):
    reason: str
    thoughts: tuple[str, ...]
    blockers: tuple[str, ...]
    screens: tuple[ScreenVisit, ...]
    actions: tuple[str, ...]


class AgentResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    success: bool
    reason: str
    answer: str
    output: T | None
    steps: int

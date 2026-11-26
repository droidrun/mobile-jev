"""Shared result contract for agent runs."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


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

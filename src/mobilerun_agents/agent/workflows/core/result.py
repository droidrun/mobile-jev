"""Shared result contract for agent runs."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class AgentResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    success: bool
    reason: str
    answer: str
    output: T | None
    steps: int

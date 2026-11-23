"""Keyword options accepted by agent workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from workflows.decorators import WorkflowGraphCheck
    from workflows.resource import ResourceManager
    from workflows.runtime.types.plugin import Runtime


class WorkflowOptionsWithoutTimeout(TypedDict, total=False):
    disable_validation: bool
    verbose: bool
    resource_manager: ResourceManager | None
    num_concurrent_runs: int | None
    runtime: Runtime | None
    workflow_name: str | None
    skip_graph_checks: set[WorkflowGraphCheck] | None


class WorkflowOptions(WorkflowOptionsWithoutTimeout, total=False):
    timeout: float | None

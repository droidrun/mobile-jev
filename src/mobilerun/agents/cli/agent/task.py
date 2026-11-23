"""Inspect and stop cloud agent tasks after disconnecting."""

import json
from pathlib import Path

import click
import httpx
from mobilerun.cli.base import render
from mobilerun.cli.base.menu import MenuGroup
from mobilerun.cli.base.run import coro
from mobilerun.cli.base.safety import confirm_action_async
from mobilerun.clients.cloud import tasks
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.errors.cloud import CloudAuthRequired, CloudResourceNotFound
from mobilerun.models.billing import Billing


@click.group(cls=MenuGroup)
def task() -> None:
    """Inspect and stop cloud agent tasks."""


def _client():
    try:
        return cloud_client(billing=Billing.NO_COMMERCE)
    except CloudAuthRequired as exc:
        raise click.ClickException("Not logged in (run `mobilerun auth login`).") from exc


def _not_found(task_id: str) -> click.ClickException:
    return click.ClickException(f"Task not found: {task_id}")


@task.command()
@click.argument("task_id")
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@coro
async def status(task_id: str, as_json: bool) -> None:
    """Show the current state and available results for TASK_ID."""
    try:
        async with _client() as client:
            result = await tasks.status(client, task_id, billing=Billing.NO_COMMERCE)
    except CloudResourceNotFound as exc:
        raise _not_found(task_id) from exc
    data = result.model_dump(mode="json", exclude_none=True)
    if as_json:
        click.echo(json.dumps(data))
    else:
        render.kv(
            (
                key.replace("_", " ").capitalize(),
                (
                    render.status(value)
                    if key in ("status", "state")
                    else render.relative_time(value) if key.endswith("_at") else value
                ),
            )
            for key, value in data.items()
        )


@task.command()
@click.argument("task_id")
@click.option("--yes", is_flag=True, help="Confirm stopping the task.")
@coro
async def stop(task_id: str, yes: bool) -> None:
    """Stop TASK_ID if it is still running."""
    await confirm_action_async("stop the task", "no commerce", "This stops the task.", yes=yes)
    try:
        async with _client() as client:
            state = await tasks.status(client, task_id, billing=Billing.NO_COMMERCE)
            if state.status in {"completed", "failed", "cancelled"}:
                click.echo(f"Task {task_id} is already finished ({state.status}).")
                return
            result = await tasks.stop(client, task_id, billing=Billing.NO_COMMERCE)
    except CloudResourceNotFound as exc:
        raise _not_found(task_id) from exc
    if result.cancelled:
        render.success(f"Stop requested for task {task_id}")
    else:
        render.warn(f"Task {task_id} was not stopped; check its status.")


@task.command()
@click.argument("task_id")
@click.option(
    "--out",
    type=click.Path(path_type=Path, file_okay=False),
    help="Save screenshots to DIR.",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON.")
@coro
async def trajectory(task_id: str, out: Path | None, as_json: bool) -> None:
    """Show TASK_ID's trajectory and optionally download screenshots."""
    try:
        async with _client() as client:
            result = await tasks.trajectory(client, task_id, billing=Billing.NO_COMMERCE)
    except CloudResourceNotFound as exc:
        raise _not_found(task_id) from exc
    data = result.model_dump(mode="json", exclude_none=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(follow_redirects=True) as http:
            for event in data["trajectory"]:
                if event.get("event") == "ScreenshotEvent":
                    screenshot = event["data"]
                    response = await http.get(screenshot["url"])
                    response.raise_for_status()
                    (out / f"screenshot-{screenshot['index']}.png").write_bytes(response.content)
    if as_json:
        click.echo(json.dumps(data))
    else:
        click.echo(json.dumps(data, indent=2))

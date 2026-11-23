import asyncio
import time

from mobilerun.agents.trajectory.writer import WriterWorker
from mobilerun.agents.workflows.droid.droid_agent import _run_finalize_stage


def test_finalize_stage_drops_a_self_cancelled_child() -> None:
    async def self_cancel() -> None:
        raise asyncio.CancelledError()

    async def run() -> None:
        completed, result = await _run_finalize_stage(
            "self-cancel", self_cancel(), time.monotonic() + 1, 1
        )
        assert (completed, result) == (False, None)

    asyncio.run(run())


def test_writer_stop_cancels_worker_when_stop_is_cancelled() -> None:
    async def run() -> None:
        worker = WriterWorker()
        worker.running = True
        worker.worker_task = asyncio.create_task(asyncio.sleep(60))
        worker.queue.put_nowait(object())
        stop_task = asyncio.create_task(worker.stop())
        await asyncio.sleep(0)
        stop_task.cancel()
        try:
            await stop_task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        assert not worker.running
        assert worker.worker_task.cancelled()

    asyncio.run(run())

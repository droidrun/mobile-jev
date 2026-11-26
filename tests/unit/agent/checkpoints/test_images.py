import asyncio

import pytest

from mobilerun_agents.agent.operators._consumers import RunConsumers
from mobilerun_agents.agent.workflows.resources import active_images
from mobilerun_agents.config.user_config import MobileConfig


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_image_store_numbers_files_without_overwriting(monkeypatch, tmp_path):
    from mobilerun_agents.agent.tools.helpers.images import ImageStore

    root = tmp_path / "images"
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(root))
    store = ImageStore("run", keep=True)
    assert not (root / "run").exists()
    first = store.save(b"first")
    second = store.save(b"second")
    third = ImageStore("run", keep=True).save(b"third")
    assert [first, second, third] == [
        root / "run/0001.png",
        root / "run/0002.png",
        root / "run/0003.png",
    ]
    assert [path.read_bytes() for path in (first, second, third)] == [
        b"first",
        b"second",
        b"third",
    ]
    assert sorted((root / "run").iterdir()) == [first, second, third]


def test_removing_a_saved_image_does_not_reuse_its_number(monkeypatch, tmp_path):
    from mobilerun_agents.agent.tools.helpers.images import ImageStore

    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(tmp_path))
    store = ImageStore("run", keep=True)
    hidden = store.save(b"screenshot to remove")
    hidden.unlink()
    visible = store.save(b"screenshot to keep")
    store.close()
    assert not hidden.exists()
    assert visible == tmp_path / "run/0002.png"
    assert visible.read_bytes() == b"screenshot to keep"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../other", "/other"])
def test_invalid_image_run_ids_cannot_target_other_folders(run_id):
    from mobilerun_agents.agent.tools.helpers.images import ImageStore

    with pytest.raises(ValueError, match="directory name"):
        ImageStore(run_id, keep=False)


@pytest.mark.anyio
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("error", [None, RuntimeError, asyncio.CancelledError])
async def test_consumers_keep_or_clean_up_run_images(monkeypatch, tmp_path, keep, error):
    config = MobileConfig.model_validate(
        {"checkpoints": {"enabled": keep}, "analytics": {"enabled": False}}
    )
    root = tmp_path / "images"
    monkeypatch.setenv("MOBILERUN_AGENT_IMAGES_DIR", str(root))
    sibling = root / "other"
    sibling.mkdir(parents=True)
    captured = asyncio.Event()

    async def run():
        async with RunConsumers(config, "run", 1):
            store = active_images.get()
            assert store is not None
            path = store.save(b"captured image")
            assert path.read_bytes() == b"captured image"
            if error is asyncio.CancelledError:
                captured.set()
                await asyncio.Future()
            if error is not None:
                raise error

    if error is None:
        await run()
    elif error is asyncio.CancelledError:
        task = asyncio.create_task(run())
        await asyncio.wait_for(captured.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(error):
            await run()
    assert (root / "run").exists() is keep
    assert (root / "run/0001.png").is_file() is keep
    assert sibling.is_dir()
    assert active_images.get() is None
